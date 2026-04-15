#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "NeuronNodeActor.generated.h"

class UStaticMeshComponent;
class UPointLightComponent;
class UTextRenderComponent;
class UMaterialInstanceDynamic;

/**
 * ANeuronNodeActor
 *
 * A single Knowledge Graph entity visualised as a glowing sphere in 3D space.
 * Each node belongs to a DSCF community (drives base color), maintains an
 * activity level (drives glow intensity / light radius), and can animate a
 * "pulse flash" when a SynapticPulse event passes through it.
 *
 * Blueprint-extensible: override OnPulseFlash, OnNeurogenesisBorn, OnPruneStart
 * to layer Niagara bursts, camera shake, sound, etc.
 *
 * Spawned and managed exclusively by ACerebrumBrain.
 */
UCLASS(Blueprintable, BlueprintType)
class CEREBRUMVISUALIZER_API ANeuronNodeActor : public AActor
{
    GENERATED_BODY()

public:
    ANeuronNodeActor();

    // ------------------------------------------------------------------
    // Identity
    // ------------------------------------------------------------------

    /** Unique entity ID from the CEREBRUM graph (e.g. "marie_curie"). */
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Cerebrum|Node")
    FString NodeId;

    /** Human-readable label shown on the billboard above the sphere. */
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Cerebrum|Node")
    FString NodeLabel;

    /** DSCF community integer ID — drives base sphere color. */
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Cerebrum|Node")
    int32 CommunityId = 0;

    // ------------------------------------------------------------------
    // Initialisation
    // ------------------------------------------------------------------

    /**
     * Called by ACerebrumBrain immediately after spawning.
     * Sets identity fields, community color, and plays the birth animation.
     */
    UFUNCTION(BlueprintCallable, Category = "Cerebrum|Node")
    void InitNode(const FString& InNodeId,
                  const FString& InLabel,
                  int32          InCommunityId,
                  FLinearColor   CommunityColor);

    // ------------------------------------------------------------------
    // Visual state setters (callable from C++ and Blueprint)
    // ------------------------------------------------------------------

    /**
     * Flash the node to indicate a synaptic pulse passing through it.
     * @param Weight  CSA attention weight [0,1] — scales flash intensity.
     * @param bIsWormhole  True if the pulse is crossing a community boundary.
     */
    UFUNCTION(BlueprintCallable, Category = "Cerebrum|Node")
    void PulseFlash(float Weight, bool bIsWormhole);

    /**
     * Set sustained glow intensity, e.g. when the community is in a
     * CorticalGlow state.
     * @param Intensity  0 = resting baseline, 1 = maximum glow.
     */
    UFUNCTION(BlueprintCallable, Category = "Cerebrum|Node")
    void SetGlowIntensity(float Intensity);

    /**
     * Tint the node to signal dissonance (orange-red alert flash).
     * Reverts to community color after DissonanceFadeDuration seconds.
     */
    UFUNCTION(BlueprintCallable, Category = "Cerebrum|Node")
    void ShowDissonance();

    /**
     * Begin a fade-out animation in preparation for destruction.
     * CerebrumBrain calls this before destroying the actor.
     */
    UFUNCTION(BlueprintCallable, Category = "Cerebrum|Node")
    void FadeOut();

    // ------------------------------------------------------------------
    // Blueprint events — override in child Blueprints for VFX / SFX
    // ------------------------------------------------------------------

    UFUNCTION(BlueprintImplementableEvent, Category = "Cerebrum|Node")
    void OnPulseFlash(float Weight, bool bIsWormhole);

    UFUNCTION(BlueprintImplementableEvent, Category = "Cerebrum|Node")
    void OnNeurogenesisBorn();

    UFUNCTION(BlueprintImplementableEvent, Category = "Cerebrum|Node")
    void OnPruneStart();

    // ------------------------------------------------------------------
    // Visual configuration (editable in Blueprint defaults)
    // ------------------------------------------------------------------

    /** Sphere radius in UE world units. */
    UPROPERTY(EditDefaultsOnly, Category = "Cerebrum|Appearance")
    float SphereRadius = 30.0f;

    /** Resting point-light radius when not active. */
    UPROPERTY(EditDefaultsOnly, Category = "Cerebrum|Appearance")
    float RestingLightRadius = 150.0f;

    /** Point-light radius at full pulse flash. */
    UPROPERTY(EditDefaultsOnly, Category = "Cerebrum|Appearance")
    float PeakLightRadius = 600.0f;

    /** Duration of a pulse flash in seconds. */
    UPROPERTY(EditDefaultsOnly, Category = "Cerebrum|Appearance")
    float PulseFlashDuration = 0.35f;

    /** Duration of fade-out before actor is destroyed. */
    UPROPERTY(EditDefaultsOnly, Category = "Cerebrum|Appearance")
    float FadeOutDuration = 1.2f;

    /** Vertical offset of the label above the sphere centre. */
    UPROPERTY(EditDefaultsOnly, Category = "Cerebrum|Appearance")
    float LabelVerticalOffset = 50.0f;

protected:
    virtual void BeginPlay() override;
    virtual void Tick(float DeltaTime) override;

private:
    // Components
    UPROPERTY(VisibleAnywhere)
    UStaticMeshComponent* SphereMesh;

    UPROPERTY(VisibleAnywhere)
    UPointLightComponent* GlowLight;

    UPROPERTY(VisibleAnywhere)
    UTextRenderComponent* LabelText;

    // Dynamic material instance for runtime color / emission control
    UPROPERTY()
    UMaterialInstanceDynamic* SphereMID;

    // Animation state
    FLinearColor BaseColor      = FLinearColor::White;
    float        FlashTimer     = 0.0f;
    float        FlashIntensity = 0.0f;
    bool         bFading        = false;
    float        FadeTimer      = 0.0f;

    // Drive material and light from current state each tick
    void UpdateVisuals(float DeltaTime);

    // Derive a stable world color from a community ID (HSV wheel)
    static FLinearColor CommunityToColor(int32 CID);
};
