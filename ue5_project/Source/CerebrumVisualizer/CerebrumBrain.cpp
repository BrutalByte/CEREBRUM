#include "CerebrumBrain.h"
#include "NeuronNodeActor.h"
#include "SynapseActor.h"
#include "CerebrumLink.h"

#include "Json.h"
#include "JsonUtilities.h"
#include "HttpModule.h"
#include "Interfaces/IHttpRequest.h"
#include "Interfaces/IHttpResponse.h"
#include "Engine/World.h"
#include "Math/UnrealMathUtility.h"

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
    CerebrumLink->OnSynapticPulse.AddDynamic(this, &ACerebrumBrain::HandleSynapticPulse);
    CerebrumLink->OnNeurogenesis.AddDynamic  (this, &ACerebrumBrain::HandleNeurogenesis);
    CerebrumLink->OnSynapticPrune.AddDynamic (this, &ACerebrumBrain::HandleSynapticPrune);
    CerebrumLink->OnCorticalGlow.AddDynamic  (this, &ACerebrumBrain::HandleCorticalGlow);
    CerebrumLink->OnDissonance.AddDynamic    (this, &ACerebrumBrain::HandleDissonance);

    if (bLoadGraphOnStart)
    {
        LoadGraphFromREST();
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

    OnGraphLoaded(NodeCount, SynapseRegistry.Num());
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
