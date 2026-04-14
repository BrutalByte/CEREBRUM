#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Websocket.h"
#include "CerebrumLink.generated.h"

// ---------------------------------------------------------------------------
// Generic catch-all delegate (backward compatible — raw JSON string)
// ---------------------------------------------------------------------------
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnNeuralEventReceived, FString, JsonData);

// ---------------------------------------------------------------------------
// Typed per-event delegates (match core/telemetry.py NeuralEventType)
// ---------------------------------------------------------------------------

/** SYNAPTIC_PULSE — a beam expansion step fires along an edge. */
DECLARE_DYNAMIC_MULTICAST_DELEGATE_SixParams(
    FOnSynapticPulse,
    FString, SourceNode,
    FString, TargetNode,
    FString, Relation,
    float,   Weight,
    int32,   HopCount,
    bool,    bIsWormhole
);

/** NEUROGENESIS — ResearchAgent created a new node in the graph. */
DECLARE_DYNAMIC_MULTICAST_DELEGATE_ThreeParams(
    FOnNeurogenesis,
    FString, NodeId,
    FString, Label,
    FString, NodeType
);

/** SYNAPTIC_PRUNE — SynapticPruner removed a synthetic edge. */
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(
    FOnSynapticPrune,
    FString, EdgeId,
    FString, Reason
);

/** CORTICAL_GLOW — a community became active during traversal. */
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(
    FOnCorticalGlow,
    FString, CommunityId,
    float,   Intensity
);

/** DISSONANCE — CerebellarEngine detected a high-score / low-consensus path. */
DECLARE_DYNAMIC_MULTICAST_DELEGATE_ThreeParams(
    FOnDissonance,
    FString, SeedEntity,
    float,   PathScore,
    float,   ConsensusScore
);

// ---------------------------------------------------------------------------
// UCerebrumLink — ActorComponent
// ---------------------------------------------------------------------------

/**
 * CerebrumLink
 *
 * Attach this component to any Actor (e.g. a GameMode or a dedicated
 * CerebrumManager actor) to receive real-time neural events from the CEREBRUM
 * WebSocket telemetry bridge (api/telemetry_bridge.py, default port 8765).
 *
 * Usage in Blueprint:
 *   1. Add CerebrumLink component to your actor.
 *   2. Call ConnectToBrain("ws://localhost:8765") from BeginPlay.
 *   3. Bind to OnSynapticPulse, OnNeurogenesis, etc.
 *   4. Spawn visual actors / Niagara effects in response to each event type.
 *
 * Usage in C++:
 *   UCerebrumLink* Link = CreateDefaultSubobject<UCerebrumLink>(TEXT("CerebrumLink"));
 *   Link->OnSynapticPulse.AddDynamic(this, &AMyActor::HandlePulse);
 *   Link->ConnectToBrain(TEXT("ws://localhost:8765"));
 */
UCLASS(ClassGroup=(Cerebrum), meta=(BlueprintSpawnableComponent))
class CEREBRUMVISUALIZER_API UCerebrumLink : public UActorComponent
{
    GENERATED_BODY()

public:
    UCerebrumLink();

    // ------------------------------------------------------------------
    // Connection management
    // ------------------------------------------------------------------

    /** Open the WebSocket connection to the CEREBRUM telemetry bridge. */
    UFUNCTION(BlueprintCallable, Category = "Cerebrum|Connection",
              meta = (ToolTip = "Connect to ws://host:port (default port 8765)"))
    void ConnectToBrain(FString URL);

    /** Close the WebSocket connection gracefully. */
    UFUNCTION(BlueprintCallable, Category = "Cerebrum|Connection")
    void Disconnect();

    /** Returns true if the WebSocket is currently connected. */
    UFUNCTION(BlueprintPure, Category = "Cerebrum|Connection")
    bool IsConnected() const;

    /** Send an arbitrary JSON string to the CEREBRUM server (future use). */
    UFUNCTION(BlueprintCallable, Category = "Cerebrum|Connection")
    void SendToBrain(FString Message);

    // ------------------------------------------------------------------
    // Delegates — bind in Blueprint or C++
    // ------------------------------------------------------------------

    /** Fires for every incoming neural event — raw JSON string. */
    UPROPERTY(BlueprintAssignable, Category = "Cerebrum|Events")
    FOnNeuralEventReceived OnNeuralEvent;

    /** Fires when a beam expansion hop occurs along a graph edge. */
    UPROPERTY(BlueprintAssignable, Category = "Cerebrum|Events")
    FOnSynapticPulse OnSynapticPulse;

    /** Fires when the ResearchAgent discovers and materialises a new node. */
    UPROPERTY(BlueprintAssignable, Category = "Cerebrum|Events")
    FOnNeurogenesis OnNeurogenesis;

    /** Fires when the SynapticPruner removes a low-utility edge. */
    UPROPERTY(BlueprintAssignable, Category = "Cerebrum|Events")
    FOnSynapticPrune OnSynapticPrune;

    /** Fires when a community becomes active during a traversal pass. */
    UPROPERTY(BlueprintAssignable, Category = "Cerebrum|Events")
    FOnCorticalGlow OnCorticalGlow;

    /** Fires when the CerebellarEngine detects a dissonant prediction. */
    UPROPERTY(BlueprintAssignable, Category = "Cerebrum|Events")
    FOnDissonance OnDissonance;

private:
    /** Underlying WebSocket managed object (null when disconnected). */
    UPROPERTY()
    UWebSocket* Socket;

    // Internal WebSocket delegate handlers
    UFUNCTION()
    void HandleConnected();

    UFUNCTION()
    void HandleConnectionError(const FString& Error);

    UFUNCTION()
    void HandleClosed(int32 StatusCode, const FString& Reason, bool bWasClean);

    UFUNCTION()
    void HandleMessage(const FString& JsonData);

    /** Parse payload and dispatch the correct typed delegate. */
    void DispatchTypedEvent(const FString& EventType,
                            const TSharedPtr<FJsonObject>& Payload);
};
