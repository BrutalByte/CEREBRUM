#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "SynapseActor.generated.h"

class UStaticMeshComponent;
class UNiagaraComponent;
class UMaterialInstanceDynamic;
class ANeuronNodeActor;

/**
 * ASynapseActor
 *
 * Visualises a directed edge between two NeuronNodeActors.
 * The body is a thin cylinder scaled and oriented between source and target
 * each tick. When a SynapticPulse event fires along this edge, a Niagara
 * particle (or fallback flash) travels from source to target.
 *
 * Weight drives opacity / emissive intensity so more confident edges
 * are visually prominent.
 *
 * ASynapseActor is spawned by ACerebrumBrain, which sets source + target via
 * SetEndpoints(). The actor self-orients and self-scales each tick.
 */
UCLASS(Blueprintable, BlueprintType)
class CEREBRUMVISUALIZER_API ASynapseActor : public AActor
{
    GENERATED_BODY()

public:
    ASynapseActor();

    // ------------------------------------------------------------------
    // Identity
    // ------------------------------------------------------------------

    /** Unique edge key — usually "SourceId::Relation::TargetId". */
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Cerebrum|Synapse")
    FString EdgeId;

    /** Relation type label (e.g. "discovered", "causes"). */
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Cerebrum|Synapse")
    FString Relation;

    /** CSA attention weight [0,1] — drives visual prominence. */
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Cerebrum|Synapse")
    float Weight = 0.5f;

    // ------------------------------------------------------------------
    // Endpoint binding
    // ------------------------------------------------------------------

    /**
     * Called by ACerebrumBrain after spawning.
     * Both pointers must be non-null; the actor will self-orient each tick.
     */
    UFUNCTION(BlueprintCallable, Category = "Cerebrum|Synapse")
    void SetEndpoints(ANeuronNodeActor* InSource,
                      ANeuronNodeActor* InTarget,
                      const FString&    InRelation,
                      float             InWeight);

    // ------------------------------------------------------------------
    // Animation
    // ------------------------------------------------------------------

    /**
     * Trigger a pulse-travel animation along the synapse from source to target.
     * @param PulseWeight  Brightness of the travelling pulse.
     * @param bIsWormhole  Cross-community edge — extra visual effect.
     */
    UFUNCTION(BlueprintCallable, Category = "Cerebrum|Synapse")
    void AnimatePulse(float PulseWeight, bool bIsWormhole);

    /**
     * Begin fade-out. CerebrumBrain calls this before destruction when a
     * SynapticPrune event fires for this edge.
     */
    UFUNCTION(BlueprintCallable, Category = "Cerebrum|Synapse")
    void FadeOut();

    // ------------------------------------------------------------------
    // Blueprint events — override for Niagara / custom VFX
    // ------------------------------------------------------------------

    /** Fires when a pulse should travel along the synapse. */
    UFUNCTION(BlueprintImplementableEvent, Category = "Cerebrum|Synapse")
    void OnPulseTravel(float PulseWeight, bool bIsWormhole);

    /** Fires when the synapse starts fading out (prune initiated). */
    UFUNCTION(BlueprintImplementableEvent, Category = "Cerebrum|Synapse")
    void OnPruneStart();

    // ------------------------------------------------------------------
    // Appearance configuration (editable in Blueprint defaults)
    // ------------------------------------------------------------------

    /** Cylinder tube radius in UU. */
    UPROPERTY(EditDefaultsOnly, Category = "Cerebrum|Appearance")
    float TubeRadius = 3.0f;

    /** Minimum opacity for low-weight edges (they dim rather than vanish). */
    UPROPERTY(EditDefaultsOnly, Category = "Cerebrum|Appearance")
    float MinOpacity = 0.15f;

    /** Duration of a pulse flash animation in seconds. */
    UPROPERTY(EditDefaultsOnly, Category = "Cerebrum|Appearance")
    float PulseDuration = 0.5f;

    /** Duration of fade-out before destruction. */
    UPROPERTY(EditDefaultsOnly, Category = "Cerebrum|Appearance")
    float FadeOutDuration = 0.8f;

    // ------------------------------------------------------------------
    // Source / target accessors (read-only in Blueprint)
    // ------------------------------------------------------------------

    UFUNCTION(BlueprintPure, Category = "Cerebrum|Synapse")
    ANeuronNodeActor* GetSourceNode() const { return SourceNode; }

    UFUNCTION(BlueprintPure, Category = "Cerebrum|Synapse")
    ANeuronNodeActor* GetTargetNode() const { return TargetNode; }

protected:
    virtual void BeginPlay() override;
    virtual void Tick(float DeltaTime) override;

private:
    // Components
    UPROPERTY(VisibleAnywhere)
    UStaticMeshComponent* CylinderMesh;

    // Dynamic material for opacity / emissive control
    UPROPERTY()
    UMaterialInstanceDynamic* CylinderMID;

    // Endpoint references (not UPROPERTY — managed by CerebrumBrain)
    UPROPERTY()
    ANeuronNodeActor* SourceNode = nullptr;

    UPROPERTY()
    ANeuronNodeActor* TargetNode  = nullptr;

    // Animation state
    float PulseTimer     = 0.0f;
    float PulseIntensity = 0.0f;
    bool  bFading        = false;
    float FadeTimer      = 0.0f;

    // Reorient and resize the cylinder to span source→target each tick
    void UpdateTransform();
    // Drive material from current pulse / fade state
    void UpdateMaterial(float DeltaTime);

    // Map relation type to a hue-offset colour for the edge tint
    static FLinearColor RelationToColor(const FString& Relation);
};
