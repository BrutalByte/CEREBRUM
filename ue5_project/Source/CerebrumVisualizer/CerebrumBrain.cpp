#include "CerebrumBrain.h"
#include "NeuronNodeActor.h"
#include "SynapseActor.h"
#include "CerebrumLink.h"
#include "CerebrumHUDOverlay.h"

#include "Json.h"
#include "JsonUtilities.h"
#include "HttpModule.h"
#include "Interfaces/IHttpRequest.h"
#include "Interfaces/IHttpResponse.h"
#include "Engine/World.h"
#include "Math/UnrealMathUtility.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

ACerebrumBrain::ACerebrumBrain()
{
    PrimaryActorTick.bCanEverTick = false;

    CerebrumLink = CreateDefaultSubobject<UCerebrumLink>(TEXT("CerebrumLink"));

    // Default spawn classes — designers can override in child Blueprint
    NeuronNodeClass  = ANeuronNodeActor::StaticClass();
    SynapseActorClass = ASynapseActor::StaticClass();
}

// ---------------------------------------------------------------------------
// BeginPlay
// ---------------------------------------------------------------------------

void ACerebrumBrain::BeginPlay()
{
    Super::BeginPlay();

    // Wire all typed delegates to our handlers
    CerebrumLink->OnSynapticPulse.AddDynamic (this, &ACerebrumBrain::HandleSynapticPulse);
    CerebrumLink->OnNeurogenesis.AddDynamic   (this, &ACerebrumBrain::HandleNeurogenesis);
    CerebrumLink->OnSynapticPrune.AddDynamic  (this, &ACerebrumBrain::HandleSynapticPrune);
    CerebrumLink->OnCorticalGlow.AddDynamic   (this, &ACerebrumBrain::HandleCorticalGlow);
    CerebrumLink->OnDissonance.AddDynamic     (this, &ACerebrumBrain::HandleDissonance);
    CerebrumLink->OnMetabolicFlux.AddDynamic  (this, &ACerebrumBrain::HandleMetabolicFlux);
    CerebrumLink->OnGUIAdaptation.AddDynamic  (this, &ACerebrumBrain::HandleGUIAdaptation);

    // Create and display HUD overlay
    if (HUDWidgetClass)
    {
        HUDWidget = CreateWidget<UUserWidget>(GetWorld(), HUDWidgetClass);
        if (HUDWidget)
        {
            HUDWidget->AddToViewport();
            if (UCerebrumHUDOverlay* Overlay = Cast<UCerebrumHUDOverlay>(HUDWidget))
            {
                Overlay->SetConnectionStatus(false, WebSocketURL);
            }
            UE_LOG(LogTemp, Log, TEXT("CerebrumBrain: HUD widget added to viewport."));
        }
    }

    if (bLoadGraphOnStart)
    {
        if (bPreferLayoutFile)
        {
            LoadGraphFromLayoutFile(); // falls back to REST internally if file absent
        }
        else
        {
            LoadGraphFromREST();
        }
    }

    ConnectToBrain();
}

// ---------------------------------------------------------------------------
// EndPlay
// ---------------------------------------------------------------------------

void ACerebrumBrain::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    CerebrumLink->Disconnect();
    Super::EndPlay(EndPlayReason);
}

// ---------------------------------------------------------------------------
// ConnectToBrain / DisconnectAndClear / ReloadGraph
// ---------------------------------------------------------------------------

void ACerebrumBrain::ConnectToBrain()
{
    CerebrumLink->ConnectToBrain(WebSocketURL);
}

void ACerebrumBrain::DisconnectAndClear()
{
    CerebrumLink->Disconnect();

    for (auto& Pair : NodeRegistry)
    {
        if (IsValid(Pair.Value)) Pair.Value->Destroy();
    }
    NodeRegistry.Empty();

    for (auto& Pair : SynapseRegistry)
    {
        if (IsValid(Pair.Value)) Pair.Value->Destroy();
    }
    SynapseRegistry.Empty();
    CommunityPositions.Empty();
    CommunityColors.Empty();
    NodeLayoutPositions.Empty();
}

void ACerebrumBrain::ReloadGraph()
{
    DisconnectAndClear();
    LoadGraphFromREST();
    ConnectToBrain();
}

// ---------------------------------------------------------------------------
// Query helpers
// ---------------------------------------------------------------------------

ANeuronNodeActor* ACerebrumBrain::FindNode(const FString& NodeId) const
{
    const ANeuronNodeActor* const* Found = NodeRegistry.Find(NodeId);
    return Found ? const_cast<ANeuronNodeActor*>(*Found) : nullptr;
}

ASynapseActor* ACerebrumBrain::FindSynapse(const FString& EdgeId) const
{
    const ASynapseActor* const* Found = SynapseRegistry.Find(EdgeId);
    return Found ? const_cast<ASynapseActor*>(*Found) : nullptr;
}

// ---------------------------------------------------------------------------
// SpawnOrGetNode
// ---------------------------------------------------------------------------

ANeuronNodeActor* ACerebrumBrain::SpawnOrGetNode(const FString& NodeId,
                                                  const FString& Label,
                                                  int32          CommunityId)
{
    if (ANeuronNodeActor** Existing = NodeRegistry.Find(NodeId))
    {
        return *Existing;
    }

    FVector Pos = ComputeNodePosition(CommunityId, NodeId);
    FActorSpawnParameters Params;
    Params.Owner = this;
    Params.SpawnCollisionHandlingOverride =
        ESpawnActorCollisionHandlingMethod::AlwaysSpawn;

    ANeuronNodeActor* Node = GetWorld()->SpawnActor<ANeuronNodeActor>(
        NeuronNodeClass, Pos, FRotator::ZeroRotator, Params);

    if (!Node) return nullptr;

    // Fetch or derive community colour
    FLinearColor* CachedColor = CommunityColors.Find(CommunityId);
    FLinearColor  Color       = CachedColor ? *CachedColor : FLinearColor::Black;

    Node->InitNode(NodeId, Label, CommunityId, Color);
    NodeRegistry.Add(NodeId, Node);
    OnNewNodeSpawned(Node);
    return Node;
}

// ---------------------------------------------------------------------------
// SpawnSynapse
// ---------------------------------------------------------------------------

ASynapseActor* ACerebrumBrain::SpawnSynapse(const FString& SrcId,
                                             const FString& TgtId,
                                             const FString& Relation,
                                             float          Weight)
{
    const FString EdgeId = FString::Printf(TEXT("%s::%s::%s"),
                                           *SrcId, *Relation, *TgtId);
    if (SynapseRegistry.Contains(EdgeId)) return SynapseRegistry[EdgeId];

    ANeuronNodeActor* SrcNode = FindNode(SrcId);
    ANeuronNodeActor* TgtNode = FindNode(TgtId);
    if (!SrcNode || !TgtNode) return nullptr;

    FActorSpawnParameters Params;
    Params.Owner = this;
    Params.SpawnCollisionHandlingOverride =
        ESpawnActorCollisionHandlingMethod::AlwaysSpawn;

    ASynapseActor* Syn = GetWorld()->SpawnActor<ASynapseActor>(
        SynapseActorClass, FVector::ZeroVector, FRotator::ZeroRotator, Params);

    if (!Syn) return nullptr;

    Syn->SetEndpoints(SrcNode, TgtNode, Relation, Weight);
    SynapseRegistry.Add(EdgeId, Syn);
    OnNewSynapseSpawned(Syn);
    return Syn;
}

// ---------------------------------------------------------------------------
// Spatial layout helpers
// ---------------------------------------------------------------------------

FVector ACerebrumBrain::GetOrCreateCommunityCenter(int32 CommunityId)
{
    if (FVector* Cached = CommunityPositions.Find(CommunityId))
    {
        return *Cached;
    }

    // Distribute community centres on a Fibonacci sphere (uniform coverage)
    // We pre-assign based on community ID so colours / positions are stable
    const int32  N = FMath::Max(CommunityPositions.Num() + 1, 1);
    const float  GoldenAngle = PI * (3.0f - FMath::Sqrt(5.0f)); // ~137.5°
    const float  Theta       = GoldenAngle * CommunityId;
    const float  Y           = 1.0f - (CommunityId / FMath::Max((float)(N * 2), 1.0f)) * 2.0f;
    const float  R           = FMath::Sqrt(FMath::Max(1.0f - Y * Y, 0.0f));

    FVector Center(
        R * FMath::Cos(Theta) * CommunityOrbitRadius,
        R * FMath::Sin(Theta) * CommunityOrbitRadius,
        Y * CommunityOrbitRadius
    );

    CommunityPositions.Add(CommunityId, Center);
    return Center;
}

FVector ACerebrumBrain::ComputeNodePosition(int32 CommunityId, const FString& NodeId)
{
    // Prefer the pre-computed position from graph_layout.json if available.
    // This gives exact Fibonacci sphere placement consistent with the Python
    // setup script, rather than re-deriving a hash-seeded random offset.
    if (const FVector* Precomputed = NodeLayoutPositions.Find(NodeId))
    {
        return *Precomputed;
    }

    FVector CommunityCenter = GetOrCreateCommunityCenter(CommunityId);

    // Deterministic random offset within the cluster sphere (seeded by NodeId)
    uint32 Seed = GetTypeHash(NodeId);
    FRandomStream RNG(Seed);

    const float Az  = RNG.FRandRange(0.0f, 2.0f * PI);
    const float El  = RNG.FRandRange(-PI / 2.0f, PI / 2.0f);
    const float R   = RNG.FRandRange(NodeClusterRadius * 0.2f, NodeClusterRadius);

    return CommunityCenter + FVector(
        R * FMath::Cos(El) * FMath::Cos(Az),
        R * FMath::Cos(El) * FMath::Sin(Az),
        R * FMath::Sin(El));
}

// ---------------------------------------------------------------------------
// REST graph load
// ---------------------------------------------------------------------------

void ACerebrumBrain::LoadGraphFromREST()
{
    // Fetch community map first (GET /communities)
    const FString URL = RESTApiBaseURL + TEXT("/communities");

    TSharedRef<IHttpRequest, ESPMode::ThreadSafe> Req =
        FHttpModule::Get().CreateRequest();
    Req->SetURL(URL);
    Req->SetVerb(TEXT("GET"));
    Req->SetHeader(TEXT("Content-Type"), TEXT("application/json"));

    if (!AuthToken.IsEmpty())
    {
        Req->SetHeader(TEXT("Authorization"),
                       FString::Printf(TEXT("Bearer %s"), *AuthToken));
    }

    Req->OnProcessRequestComplete().BindLambda(
        [this](FHttpRequestPtr Request,
               FHttpResponsePtr Response,
               bool bSuccess)
        {
            if (!bSuccess || !Response.IsValid() ||
                Response->GetResponseCode() != 200)
            {
                UE_LOG(LogTemp, Warning,
                       TEXT("CerebrumBrain: /communities request failed. "
                            "Visualisation starts empty."));
                return;
            }
            ParseGraphPayload(Response->GetContentAsString());
        });

    Req->ProcessRequest();
}

void ACerebrumBrain::ParseGraphPayload(const FString& JsonBody)
{
    TSharedPtr<FJsonObject> Root;
    TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(JsonBody);
    if (!FJsonSerializer::Deserialize(Reader, Root) || !Root.IsValid())
    {
        UE_LOG(LogTemp, Warning,
               TEXT("CerebrumBrain: Could not parse /communities JSON."));
        return;
    }

    // Expected: { "communities": { "node_id": community_int, ... } }
    const TSharedPtr<FJsonObject>* CommObj = nullptr;
    if (!Root->TryGetObjectField(TEXT("communities"), CommObj) || !CommObj)
    {
        return;
    }

    for (const auto& KV : (*CommObj)->Values)
    {
        const FString& NodeId = KV.Key;
        int32 CID = 0;
        if (KV.Value->TryGetNumber(CID))
        {
            SpawnOrGetNode(NodeId, NodeId, CID);
        }
    }

    // Now try to fetch edges for a seed entity to pre-populate some synapses.
    // A full edge preload requires a dedicated endpoint; for now just log readiness.
    const int32 NodeCount = NodeRegistry.Num();
    UE_LOG(LogTemp, Log,
           TEXT("CerebrumBrain: Loaded %d nodes from /communities."), NodeCount);

    if (UCerebrumHUDOverlay* Overlay = Cast<UCerebrumHUDOverlay>(HUDWidget))
    {
        Overlay->UpdateNodeCount(NodeCount, SynapseRegistry.Num());
        Overlay->SetConnectionStatus(true, WebSocketURL);
    }
    OnGraphLoaded(NodeCount, SynapseRegistry.Num());
}

// ---------------------------------------------------------------------------
// Layout-file graph load
// ---------------------------------------------------------------------------

void ACerebrumBrain::LoadGraphFromLayoutFile()
{
    // Resolve path relative to the project Content directory
    const FString FullPath = FPaths::Combine(
        FPaths::ProjectContentDir(), GraphLayoutFilePath);

    FString JsonBody;
    if (!FFileHelper::LoadFileToString(JsonBody, *FullPath))
    {
        UE_LOG(LogTemp, Warning,
               TEXT("CerebrumBrain: Layout file not found at '%s'. "
                    "Falling back to REST /communities."), *FullPath);
        LoadGraphFromREST();
        return;
    }

    UE_LOG(LogTemp, Log,
           TEXT("CerebrumBrain: Loading graph layout from '%s'."), *FullPath);

    if (!ParseLayoutPayload(JsonBody))
    {
        UE_LOG(LogTemp, Warning,
               TEXT("CerebrumBrain: Failed to parse layout file '%s'. "
                    "Falling back to REST /communities."), *FullPath);
        LoadGraphFromREST();
    }
}

bool ACerebrumBrain::ParseLayoutPayload(const FString& JsonBody)
{
    TSharedPtr<FJsonObject> Root;
    TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(JsonBody);
    if (!FJsonSerializer::Deserialize(Reader, Root) || !Root.IsValid())
    {
        UE_LOG(LogTemp, Warning,
               TEXT("CerebrumBrain: Could not deserialise layout JSON."));
        return false;
    }

    // ── 1. Parse community metadata: populate CommunityPositions + CommunityColors ──
    const TArray<TSharedPtr<FJsonValue>>* CommArray = nullptr;
    if (Root->TryGetArrayField(TEXT("communities"), CommArray) && CommArray)
    {
        for (const TSharedPtr<FJsonValue>& Val : *CommArray)
        {
            const TSharedPtr<FJsonObject>& Comm = Val->AsObject();
            if (!Comm.IsValid()) continue;

            int32 CID = 0;
            double Tmp = 0.0;
            if (Comm->TryGetNumberField(TEXT("community_id"), Tmp))
                CID = static_cast<int32>(Tmp);

            // Community center
            const TSharedPtr<FJsonObject>* CenterObj = nullptr;
            if (Comm->TryGetObjectField(TEXT("center"), CenterObj) && CenterObj)
            {
                double X = 0, Y = 0, Z = 0;
                (*CenterObj)->TryGetNumberField(TEXT("x"), X);
                (*CenterObj)->TryGetNumberField(TEXT("y"), Y);
                (*CenterObj)->TryGetNumberField(TEXT("z"), Z);
                CommunityPositions.Add(CID, FVector(
                    static_cast<float>(X),
                    static_cast<float>(Y),
                    static_cast<float>(Z)));
            }

            // Community color
            const TSharedPtr<FJsonObject>* ColorObj = nullptr;
            if (Comm->TryGetObjectField(TEXT("color"), ColorObj) && ColorObj)
            {
                double R = 0, G = 0, B = 0;
                (*ColorObj)->TryGetNumberField(TEXT("r"), R);
                (*ColorObj)->TryGetNumberField(TEXT("g"), G);
                (*ColorObj)->TryGetNumberField(TEXT("b"), B);
                CommunityColors.Add(CID, FLinearColor(
                    static_cast<float>(R),
                    static_cast<float>(G),
                    static_cast<float>(B)));
            }
        }
    }

    // ── 2. Parse node entries: cache positions, then spawn ──────────────────
    const TArray<TSharedPtr<FJsonValue>>* NodesArray = nullptr;
    if (!Root->TryGetArrayField(TEXT("nodes"), NodesArray) || !NodesArray)
    {
        UE_LOG(LogTemp, Warning,
               TEXT("CerebrumBrain: Layout JSON has no 'nodes' array."));
        return false;
    }

    int32 SpawnedCount = 0;
    for (const TSharedPtr<FJsonValue>& Val : *NodesArray)
    {
        const TSharedPtr<FJsonObject>& Node = Val->AsObject();
        if (!Node.IsValid()) continue;

        FString NodeId, Label;
        Node->TryGetStringField(TEXT("node_id"), NodeId);
        Node->TryGetStringField(TEXT("label"),   Label);
        if (Label.IsEmpty()) Label = NodeId;

        double CIDDouble = 0.0;
        int32  CID = 0;
        if (Node->TryGetNumberField(TEXT("community_id"), CIDDouble))
            CID = static_cast<int32>(CIDDouble);

        // Cache exact pre-computed position
        const TSharedPtr<FJsonObject>* PosObj = nullptr;
        if (Node->TryGetObjectField(TEXT("position"), PosObj) && PosObj)
        {
            double X = 0, Y = 0, Z = 0;
            (*PosObj)->TryGetNumberField(TEXT("x"), X);
            (*PosObj)->TryGetNumberField(TEXT("y"), Y);
            (*PosObj)->TryGetNumberField(TEXT("z"), Z);
            NodeLayoutPositions.Add(NodeId, FVector(
                static_cast<float>(X),
                static_cast<float>(Y),
                static_cast<float>(Z)));
        }

        SpawnOrGetNode(NodeId, Label, CID);
        ++SpawnedCount;
    }

    UE_LOG(LogTemp, Log,
           TEXT("CerebrumBrain: Layout file loaded — %d nodes, %d communities."),
           SpawnedCount, CommunityPositions.Num());

    // ── 3. Spawn synapses from edges[] array (layout version 1.1+) ──────────
    int32 SynapseCount = 0;
    const TArray<TSharedPtr<FJsonValue>>* EdgesArray = nullptr;
    if (Root->TryGetArrayField(TEXT("edges"), EdgesArray) && EdgesArray)
    {
        for (const TSharedPtr<FJsonValue>& Val : *EdgesArray)
        {
            const TSharedPtr<FJsonObject>& EdgeObj = Val->AsObject();
            if (!EdgeObj.IsValid()) continue;

            FString SrcId, TgtId, Relation;
            EdgeObj->TryGetStringField(TEXT("source_id"),    SrcId);
            EdgeObj->TryGetStringField(TEXT("target_id"),    TgtId);
            EdgeObj->TryGetStringField(TEXT("relation_type"), Relation);
            double W = 1.0;
            EdgeObj->TryGetNumberField(TEXT("weight"), W);

            if (SrcId.IsEmpty() || TgtId.IsEmpty()) continue;

            // Ensure both endpoint nodes exist before trying to spawn the synapse
            if (!NodeRegistry.Contains(SrcId) || !NodeRegistry.Contains(TgtId))
                continue;

            if (SpawnSynapse(SrcId, TgtId, Relation, static_cast<float>(W)))
            {
                ++SynapseCount;
            }
        }

        UE_LOG(LogTemp, Log,
               TEXT("CerebrumBrain: Spawned %d synapses from layout edges."),
               SynapseCount);
    }

    if (UCerebrumHUDOverlay* Overlay = Cast<UCerebrumHUDOverlay>(HUDWidget))
    {
        Overlay->UpdateNodeCount(SpawnedCount, SynapseCount);
        Overlay->SetConnectionStatus(true, WebSocketURL);
    }
    OnGraphLoaded(SpawnedCount, SynapseCount);
    return SpawnedCount > 0;
}

// ---------------------------------------------------------------------------
// Event handlers (wired to UCerebrumLink delegates)
// ---------------------------------------------------------------------------

void ACerebrumBrain::HandleSynapticPulse(FString SourceNode, FString TargetNode,
                                          FString Relation, float Weight,
                                          int32 HopCount, bool bIsWormhole)
{
    // Ensure both nodes exist (neurogenesis may have been missed)
    SpawnOrGetNode(SourceNode, SourceNode, 0);
    SpawnOrGetNode(TargetNode, TargetNode, 0);

    // Find or spawn the synapse
    const FString EdgeId = FString::Printf(TEXT("%s::%s::%s"),
                                           *SourceNode, *Relation, *TargetNode);
    ASynapseActor* Syn = SynapseRegistry.FindRef(EdgeId);
    if (!Syn)
    {
        Syn = SpawnSynapse(SourceNode, TargetNode, Relation, Weight);
    }
    if (Syn)
    {
        Syn->AnimatePulse(Weight, bIsWormhole);
    }
}

void ACerebrumBrain::HandleNeurogenesis(FString NodeId, FString Label,
                                         FString NodeType)
{
    // NodeType hint: could be "entity", "synthesis", "bridge_twin" etc.
    // Default to community 0 until we receive a CorticalGlow that reveals
    // the community — or a future endpoint provides it.
    SpawnOrGetNode(NodeId, Label.IsEmpty() ? NodeId : Label, 0);
}

void ACerebrumBrain::HandleSynapticPrune(FString EdgeId, FString Reason)
{
    if (ASynapseActor** Found = SynapseRegistry.Find(EdgeId))
    {
        if (IsValid(*Found))
        {
            (*Found)->FadeOut();     // self-destructs after fade
        }
        SynapseRegistry.Remove(EdgeId);
    }
}

void ACerebrumBrain::HandleCorticalGlow(FString CommunityId, float Intensity)
{
    // CommunityId is a string from the JSON; convert to int
    const int32 CID = FCString::Atoi(*CommunityId);

    for (auto& Pair : NodeRegistry)
    {
        if (IsValid(Pair.Value) && Pair.Value->CommunityId == CID)
        {
            Pair.Value->SetGlowIntensity(Intensity);
        }
    }
}

void ACerebrumBrain::HandleDissonance(FString SeedEntity, float PathScore,
                                       float ConsensusScore)
{
    if (ANeuronNodeActor* Node = FindNode(SeedEntity))
    {
        Node->ShowDissonance();
    }
    OnDissonanceAlert(SeedEntity, PathScore, ConsensusScore);
}

void ACerebrumBrain::HandleMetabolicFlux(float Reinforcement, float Arousal,
                                          float Novelty, float Cohesion,
                                          float Persistence, float LearningRateScale)
{
    if (UCerebrumHUDOverlay* Overlay = Cast<UCerebrumHUDOverlay>(HUDWidget))
    {
        Overlay->UpdateMetabolics(Reinforcement, Arousal, Novelty, Cohesion, Persistence);
    }
    OnMetabolicUpdate(Reinforcement, Arousal, Novelty, Cohesion, Persistence, LearningRateScale);
}

void ACerebrumBrain::HandleGUIAdaptation(FString Action, FString Target, FString DataJson)
{
    UE_LOG(LogTemp, Log, TEXT("CerebrumBrain: GUI Adaptation - Action: %s, Target: %s"), *Action, *Target);

    if (Action == TEXT("layout_shift"))
    {
        // Target is the layout scheme: "clustered", "neighborhood", "hierarchical", "force"
        // DataJson contains "focal_node" if scheme is neighborhood

        TSharedPtr<FJsonObject> Payload;
        TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(DataJson);
        FString FocalNode;
        if (FJsonSerializer::Deserialize(Reader, Payload) && Payload.IsValid())
        {
            Payload->TryGetStringField(TEXT("focal_node"), FocalNode);
        }

        if (Target == TEXT("neighborhood") && !FocalNode.IsEmpty())
        {
            // --- Neighborhood Layout (Layered Ego-Network) ---
            ANeuronNodeActor* AnchorNode = FindNode(FocalNode);
            if (AnchorNode)
            {
                AnchorNode->SetActorLocation(FVector::ZeroVector);
            }
            // Trigger Blueprint event for more complex transition/layout logic
        }
        else if (Target == TEXT("clustered"))
        {
            // Reset to default community layout
            for (auto& Pair : NodeRegistry)
            {
                if (IsValid(Pair.Value))
                {
                    FVector NewPos = ComputeNodePosition(Pair.Value->GetCommunityId(), Pair.Key);
                    Pair.Value->SetActorLocation(NewPos);
                }
            }
        }
    }

    // Forward to Blueprint — child BP calls widget functions (SetVisibility, etc.)
    OnGUIAdaptationEvent(Action, Target, DataJson);
}
