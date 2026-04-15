#include "SynapseActor.h"
#include "NeuronNodeActor.h"

#include "Components/StaticMeshComponent.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "UObject/ConstructorHelpers.h"
#include "Engine/StaticMesh.h"
#include "Math/UnrealMathUtility.h"

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

ASynapseActor::ASynapseActor()
{
    PrimaryActorTick.bCanEverTick = true;

    // Root is the cylinder mesh
    CylinderMesh = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("CylinderMesh"));
    RootComponent = CylinderMesh;

    static ConstructorHelpers::FObjectFinder<UStaticMesh> CylAsset(
        TEXT("/Engine/BasicShapes/Cylinder.Cylinder"));
    if (CylAsset.Succeeded())
    {
        CylinderMesh->SetStaticMesh(CylAsset.Object);
    }
    CylinderMesh->SetCollisionEnabled(ECollisionEnabled::NoCollision);
    CylinderMesh->SetCastShadow(false);
}

// ---------------------------------------------------------------------------
// BeginPlay
// ---------------------------------------------------------------------------

void ASynapseActor::BeginPlay()
{
    Super::BeginPlay();

    UMaterialInterface* BaseMat = CylinderMesh->GetMaterial(0);
    if (BaseMat)
    {
        CylinderMID = UMaterialInstanceDynamic::Create(BaseMat, this);
        CylinderMesh->SetMaterial(0, CylinderMID);
    }

    // Initial weight-based opacity
    if (CylinderMID)
    {
        const float Opacity = FMath::Lerp(MinOpacity, 0.7f, Weight);
        CylinderMID->SetScalarParameterValue(TEXT("Opacity"), Opacity);
        CylinderMID->SetVectorParameterValue(TEXT("BaseColor"),
                                             RelationToColor(Relation));
        CylinderMID->SetScalarParameterValue(TEXT("EmissiveIntensity"), 0.1f);
    }
}

// ---------------------------------------------------------------------------
// SetEndpoints
// ---------------------------------------------------------------------------

void ASynapseActor::SetEndpoints(ANeuronNodeActor* InSource,
                                  ANeuronNodeActor* InTarget,
                                  const FString&    InRelation,
                                  float             InWeight)
{
    SourceNode = InSource;
    TargetNode  = InTarget;
    Relation    = InRelation;
    Weight      = FMath::Clamp(InWeight, 0.0f, 1.0f);

    // EdgeId: "source::relation::target"
    EdgeId = FString::Printf(TEXT("%s::%s::%s"),
                             *InSource->NodeId, *InRelation, *InTarget->NodeId);

    if (CylinderMID)
    {
        CylinderMID->SetVectorParameterValue(TEXT("BaseColor"),
                                             RelationToColor(Relation));
        const float Opacity = FMath::Lerp(MinOpacity, 0.7f, Weight);
        CylinderMID->SetScalarParameterValue(TEXT("Opacity"), Opacity);
    }

    // Orient immediately so it doesn't appear at origin for one frame
    UpdateTransform();
}

// ---------------------------------------------------------------------------
// AnimatePulse
// ---------------------------------------------------------------------------

void ASynapseActor::AnimatePulse(float PulseWeight, bool bIsWormhole)
{
    PulseIntensity = FMath::Clamp(PulseWeight, 0.2f, 1.0f) * (bIsWormhole ? 1.6f : 1.0f);
    PulseTimer     = PulseDuration;

    // Notify both endpoint nodes so they flash in sync
    if (SourceNode) SourceNode->PulseFlash(PulseWeight, bIsWormhole);
    if (TargetNode)  TargetNode->PulseFlash(PulseWeight * 0.75f, bIsWormhole);

    OnPulseTravel(PulseWeight, bIsWormhole);
}

// ---------------------------------------------------------------------------
// FadeOut
// ---------------------------------------------------------------------------

void ASynapseActor::FadeOut()
{
    bFading   = true;
    FadeTimer = FadeOutDuration;
    OnPruneStart();
}

// ---------------------------------------------------------------------------
// Tick
// ---------------------------------------------------------------------------

void ASynapseActor::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);

    // Re-orient each tick so the tube stays connected even if nodes move
    if (SourceNode && TargetNode && !bFading)
    {
        UpdateTransform();
    }

    UpdateMaterial(DeltaTime);
}

// ---------------------------------------------------------------------------
// UpdateTransform
// ---------------------------------------------------------------------------

void ASynapseActor::UpdateTransform()
{
    if (!SourceNode || !TargetNode) return;

    const FVector SrcLoc = SourceNode->GetActorLocation();
    const FVector TgtLoc = TargetNode->GetActorLocation();
    const FVector Mid    = (SrcLoc + TgtLoc) * 0.5f;
    const FVector Delta  = TgtLoc - SrcLoc;
    const float   Length = Delta.Size();

    if (Length < KINDA_SMALL_NUMBER) return;

    // Position at midpoint
    SetActorLocation(Mid);

    // Orient: UE cylinder's long axis is Z; rotate Z to align with Delta
    const FVector UpDir = Delta.GetSafeNormal();
    const FQuat   Rot   = FQuat::FindBetweenNormals(FVector::UpVector, UpDir);
    SetActorRotation(Rot);

    // Scale: X/Y = tube radius (default mesh radius ~50 UU), Z = half-length
    const float RadiusScale = TubeRadius / 50.0f;
    const float LengthScale = (Length * 0.5f) / 100.0f; // mesh default height ~100 UU
    CylinderMesh->SetRelativeScale3D(FVector(RadiusScale, RadiusScale, LengthScale));
}

// ---------------------------------------------------------------------------
// UpdateMaterial
// ---------------------------------------------------------------------------

void ASynapseActor::UpdateMaterial(float DeltaTime)
{
    if (!CylinderMID) return;

    // --- Pulse flash decay ---
    if (PulseTimer > 0.0f)
    {
        PulseTimer -= DeltaTime;
        const float T = FMath::Max(PulseTimer / PulseDuration, 0.0f);
        CylinderMID->SetScalarParameterValue(TEXT("EmissiveIntensity"),
                                             FMath::Lerp(0.1f, 3.0f * PulseIntensity, T));
    }

    // --- Fade-out ---
    if (bFading)
    {
        FadeTimer -= DeltaTime;
        const float Alpha = FMath::Max(FadeTimer / FadeOutDuration, 0.0f);
        CylinderMID->SetScalarParameterValue(TEXT("Opacity"), Alpha * 0.7f);
        CylinderMID->SetScalarParameterValue(TEXT("EmissiveIntensity"), Alpha * 0.1f);

        if (FadeTimer <= 0.0f)
        {
            Destroy();
        }
    }
}

// ---------------------------------------------------------------------------
// RelationToColor  (static helper)
// ---------------------------------------------------------------------------

FLinearColor ASynapseActor::RelationToColor(const FString& Relation)
{
    // Simple hash: sum char values mod 256, map to hue
    uint32 Hash = 5381;
    for (TCHAR C : Relation)
    {
        Hash = ((Hash << 5) + Hash) + static_cast<uint32>(C);
    }
    const float Hue = (Hash % 256) / 255.0f;
    return FLinearColor::MakeFromHSV8(
        static_cast<uint8>(Hue * 255),
        160,    // Saturation — slightly desaturated so nodes stand out
        200);   // Value
}
