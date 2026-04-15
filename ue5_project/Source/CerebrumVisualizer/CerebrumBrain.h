#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "CerebrumLink.h"
#include "CerebrumBrain.generated.h"

class ANeuronNodeActor;
class ASynapseActor;
class UCerebrumLink;

// ---------------------------------------------------------------------------
// JSON graph layout entry — used when loading initial state from REST
// ---------------------------------------------------------------------------

USTRUCT(BlueprintType)
struct FCerebrumNodeLayout
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly) FString NodeId;
    UPROPERTY(BlueprintReadOnly) FString Label;
    UPROPERTY(BlueprintReadOnly) int32   CommunityId = 0;
    UPROPERTY(BlueprintReadOnly) FVector WorldPosition = FVector::ZeroVector;
};

USTRUCT(BlueprintType)
struct FCerebrumEdgeLayout
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly) FString SourceId;
    UPROPERTY(BlueprintReadOnly) FString TargetId;
    UPROPERTY(BlueprintReadOnly) FString Relation;
    UPROPERTY(BlueprintReadOnly) float   Weight = 0.5f;
};

// ---------------------------------------------------------------------------
// ACerebrumBrain
// ---------------------------------------------------------------------------

/**
 * ACerebrumBrain
 *
 * The single orchestrator actor for the CEREBRUM 3D visualisation.
 * Responsibilities:
 *
 *  1. Connects to the CEREBRUM telemetry WebSocket via UCerebrumLink.
 *  2. Optionally loads the initial graph topology from the REST API on BeginPlay
 *     and spawns ANeuronNodeActor + ASynapseActor for every entity/edge.
 *  3. Responds to real-time events:
 *       SYNAPTIC_PULSE   → AnimatePulse on the matching ASynapseActor
 *       NEUROGENESIS     → Spawn a new ANeuronNodeActor
 *       SYNAPTIC_PRUNE   → FadeOut + destroy the matching ASynapseActor
 *       CORTICAL_GLOW    → SetGlowIntensity on all nodes in that community
 *       DISSONANCE       → ShowDissonance on the seed node
 *  4. Manages spatial layout:
 *       - Communities are arranged on a sphere of radius CommunityOrbitRadius.
 *       - Nodes within each community cluster around their community centre
 *         with radius NodeClusterRadius.
 *       - New neurogenesis nodes are placed near their community centre.
 *
 * Place exactly one CerebrumBrain actor in your level.
 * Set WebSocketURL to "ws://localhost:8765" (or your CEREBRUM host).
 * Set RESTApiBaseURL to "http://localhost:8200" to enable graph pre-load.
 *
 * The actor also exposes Blueprint hooks for every event so designers can
 * layer camera shake, post-process effects, ambient audio etc.
 */
UCLASS(Blueprintable, BlueprintType)
class CEREBRUMVISUALIZER_API ACerebrumBrain : public AActor
{
    GENERATED_BODY()

public:
    ACerebrumBrain();

    // ------------------------------------------------------------------
    // Configuration (set in Blueprint / Details panel)
    // ------------------------------------------------------------------

    /** WebSocket URL for the CEREBRUM telemetry bridge. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cerebrum|Connection")
    FString WebSocketURL = TEXT("ws://localhost:8765");

    /** Base URL for CEREBRUM REST API used to pre-load graph topology. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cerebrum|Connection")
    FString RESTApiBaseURL = TEXT("http://localhost:8200");

    /** JWT token for REST authentication (leave empty for anonymous mode). */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cerebrum|Connection")
    FString AuthToken;

    /** If true, load initial graph from REST API on BeginPlay. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cerebrum|Startup")
    bool bLoadGraphOnStart = true;

    /** Radius (UU) of the sphere on which community centres are placed. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cerebrum|Layout")
    float CommunityOrbitRadius = 2500.0f;

    /** Radius (UU) of the cluster around each community centre. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cerebrum|Layout")
    float NodeClusterRadius = 600.0f;

    /** Actor class to spawn for nodes — subclass to add custom VFX. */
    UPROPERTY(EditDefaultsOnly, Category = "Cerebrum|Spawning")
    TSubclassOf<ANeuronNodeActor> NeuronNodeClass;

    /** Actor class to spawn for edges — subclass for custom VFX. */
    UPROPERTY(EditDefaultsOnly, Category = "Cerebrum|Spawning")
    TSubclassOf<ASynapseActor> SynapseActorClass;

    // ------------------------------------------------------------------
    // Public API (callable from Blueprint)
    // ------------------------------------------------------------------

    /** Immediately connect the WebSocket (called automatically on BeginPlay). */
    UFUNCTION(BlueprintCallable, Category = "Cerebrum|Connection")
    void ConnectToBrain();

    /** Disconnect and clear all visual actors. */
    UFUNCTION(BlueprintCallable, Category = "Cerebrum|Connection")
    void DisconnectAndClear();

    /** Force a REST graph reload (useful after hot-reloading the server). */
    UFUNCTION(BlueprintCallable, Category = "Cerebrum|Startup")
    void ReloadGraph();

    /** Returns the node actor for a given entity ID, or null if not spawned. */
    UFUNCTION(BlueprintPure, Category = "Cerebrum|Query")
    ANeuronNodeActor* FindNode(const FString& NodeId) const;

    /** Returns the synapse actor for a given edge key, or null if not spawned. */
    UFUNCTION(BlueprintPure, Category = "Cerebrum|Query")
    ASynapseActor* FindSynapse(const FString& EdgeId) const;

    /** Number of spawned node actors. */
    UFUNCTION(BlueprintPure, Category = "Cerebrum|Query")
    int32 GetNodeCount() const { return NodeRegistry.Num(); }

    /** Number of spawned synapse actors. */
    UFUNCTION(BlueprintPure, Category = "Cerebrum|Query")
    int32 GetSynapseCount() const { return SynapseRegistry.Num(); }

    // ------------------------------------------------------------------
    // Blueprint events — override for level-wide VFX / audio
    // ------------------------------------------------------------------

    UFUNCTION(BlueprintImplementableEvent, Category = "Cerebrum|Events")
    void OnGraphLoaded(int32 NodeCount, int32 EdgeCount);

    UFUNCTION(BlueprintImplementableEvent, Category = "Cerebrum|Events")
    void OnNewNodeSpawned(ANeuronNodeActor* Node);

    UFUNCTION(BlueprintImplementableEvent, Category = "Cerebrum|Events")
    void OnNewSynapseSpawned(ASynapseActor* Synapse);

    UFUNCTION(BlueprintImplementableEvent, Category = "Cerebrum|Events")
    void OnDissonanceAlert(const FString& SeedEntity, float PathScore, float ConsensusScore);

protected:
    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

private:
    // ------------------------------------------------------------------
    // Components
    // ------------------------------------------------------------------

    UPROPERTY(VisibleAnywhere)
    UCerebrumLink* CerebrumLink;

    // ------------------------------------------------------------------
    // Live registries
    // ------------------------------------------------------------------

    /** NodeId → spawned actor */
    UPROPERTY()
    TMap<FString, ANeuronNodeActor*> NodeRegistry;

    /** EdgeId ("src::rel::tgt") → spawned actor */
    UPROPERTY()
    TMap<FString, ASynapseActor*> SynapseRegistry;

    /** CommunityId → community centre world position */
    TMap<int32, FVector> CommunityPositions;

    /** CommunityId → colour */
    TMap<int32, FLinearColor> CommunityColors;

    // ------------------------------------------------------------------
    // Internal helpers — implemented in CerebrumBrain.cpp
    // ------------------------------------------------------------------

    // Spawn / find
    ANeuronNodeActor* SpawnOrGetNode(const FString& NodeId,
                                     const FString& Label,
                                     int32          CommunityId);

    ASynapseActor* SpawnSynapse(const FString& SrcId,
                                const FString& TgtId,
                                const FString& Relation,
                                float          Weight);

    FVector ComputeNodePosition(int32 CommunityId, const FString& NodeId);
    FVector GetOrCreateCommunityCenter(int32 CommunityId);

    // REST graph load
    void LoadGraphFromREST();
    void ParseGraphPayload(const FString& JsonBody);

    // CerebrumLink event handlers
    UFUNCTION()
    void HandleSynapticPulse(FString SourceNode, FString TargetNode,
                              FString Relation, float Weight,
                              int32 HopCount, bool bIsWormhole);

    UFUNCTION()
    void HandleNeurogenesis(FString NodeId, FString Label, FString NodeType);

    UFUNCTION()
    void HandleSynapticPrune(FString EdgeId, FString Reason);

    UFUNCTION()
    void HandleCorticalGlow(FString CommunityId, float Intensity);

    UFUNCTION()
    void HandleDissonance(FString SeedEntity, float PathScore, float ConsensusScore);
};
