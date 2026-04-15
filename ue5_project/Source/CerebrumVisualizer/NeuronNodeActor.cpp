#include "NeuronNodeActor.h"

#include "Components/StaticMeshComponent.h"
#include "Components/PointLightComponent.h"
#include "Components/TextRenderComponent.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "UObject/ConstructorHelpers.h"
#include "Engine/StaticMesh.h"
#include "Math/UnrealMathUtility.h"

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

ANeuronNodeActor::ANeuronNodeActor()
{
    PrimaryActorTick.bCanEverTick = true;

    // Root — sphere mesh
    SphereMesh = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("SphereMesh"));
    RootComponent = SphereMesh;

    static ConstructorHelpers::FObjectFinder<UStaticMesh> SphereMeshAsset(
        TEXT("/Engine/BasicShapes/Sphere.Sphere"));
    if (SphereMeshAsset.Succeeded())
    {
        SphereMesh->SetStaticMesh(SphereMeshAsset.Object);
    }
    SphereMesh->SetCollisionEnabled(ECollisionEnabled::NoCollision);

    // Point light for glow
    GlowLight = CreateDefaultSubobject<UPointLightComponent>(TEXT("GlowLight"));
    GlowLight->SetupAttachment(RootComponent);
    GlowLight->SetIntensity(1500.0f);
    GlowLight->SetAttenuationRadius(RestingLightRadius);
    GlowLight->SetCastShadows(false);

    // Billboard label
    LabelText = CreateDefaultSubobject<UTextRenderComponent>(TEXT("LabelText"));
    LabelText->SetupAttachment(RootComponent);
    LabelText->SetRelativeLocation(FVector(0.0f, 0.0f, LabelVerticalOffset));
    LabelText->SetHorizontalAlignment(EHTA_Center);
    LabelText->SetWorldSize(20.0f);
    LabelText->SetTextRenderColor(FColor::White);
    LabelText->SetText(FText::FromString(TEXT("")));
}

// ---------------------------------------------------------------------------
// BeginPlay
// ---------------------------------------------------------------------------

void ANeuronNodeActor::BeginPlay()
{
    Super::BeginPlay();

    // Scale sphere mesh so radius matches SphereRadius
    // UE default sphere mesh has radius ~50 UU, so scale = SphereRadius / 50
    const float Scale = SphereRadius / 50.0f;
    SphereMesh->SetRelativeScale3D(FVector(Scale));

    // Create dynamic material instance for runtime colour control
    UMaterialInterface* BaseMat = SphereMesh->GetMaterial(0);
    if (BaseMat)
    {
        SphereMID = UMaterialInstanceDynamic::Create(BaseMat, this);
        SphereMesh->SetMaterial(0, SphereMID);
    }
}

// ---------------------------------------------------------------------------
// Tick
// ---------------------------------------------------------------------------

void ANeuronNodeActor::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);
    UpdateVisuals(DeltaTime);
}

// ---------------------------------------------------------------------------
// InitNode
// ---------------------------------------------------------------------------

void ANeuronNodeActor::InitNode(const FString& InNodeId,
                                const FString& InLabel,
                                int32          InCommunityId,
                                FLinearColor   CommunityColor)
{
    NodeId      = InNodeId;
    NodeLabel   = InLabel;
    CommunityId = InCommunityId;
    BaseColor   = (CommunityColor == FLinearColor::Black)
                      ? CommunityToColor(InCommunityId)
                      : CommunityColor;

    LabelText->SetText(FText::FromString(InLabel));
    GlowLight->SetLightColor(BaseColor);

    if (SphereMID)
    {
        SphereMID->SetVectorParameterValue(TEXT("BaseColor"), BaseColor);
        SphereMID->SetScalarParameterValue(TEXT("EmissiveIntensity"), 0.3f);
    }

    // Play birth event for Blueprint VFX / sound
    OnNeurogenesisBorn();
}

// ---------------------------------------------------------------------------
// PulseFlash
// ---------------------------------------------------------------------------

void ANeuronNodeActor::PulseFlash(float Weight, bool bIsWormhole)
{
    // Weight [0,1] drives flash intensity; wormhole → extra bright
    FlashIntensity = FMath::Clamp(Weight, 0.2f, 1.0f) * (bIsWormhole ? 1.5f : 1.0f);
    FlashTimer     = PulseFlashDuration;

    OnPulseFlash(Weight, bIsWormhole);
}

// ---------------------------------------------------------------------------
// SetGlowIntensity
// ---------------------------------------------------------------------------

void ANeuronNodeActor::SetGlowIntensity(float Intensity)
{
    const float Clamped = FMath::Clamp(Intensity, 0.0f, 1.0f);
    GlowLight->SetIntensity(FMath::Lerp(500.0f, 5000.0f, Clamped));
    GlowLight->SetAttenuationRadius(
        FMath::Lerp(RestingLightRadius, PeakLightRadius, Clamped));

    if (SphereMID)
    {
        SphereMID->SetScalarParameterValue(TEXT("EmissiveIntensity"),
                                           FMath::Lerp(0.3f, 2.5f, Clamped));
    }
}

// ---------------------------------------------------------------------------
// ShowDissonance
// ---------------------------------------------------------------------------

void ANeuronNodeActor::ShowDissonance()
{
    // Orange-red flash — temporarily override base colour
    if (SphereMID)
    {
        SphereMID->SetVectorParameterValue(TEXT("BaseColor"),
                                           FLinearColor(1.0f, 0.35f, 0.05f));
        SphereMID->SetScalarParameterValue(TEXT("EmissiveIntensity"), 2.0f);
    }
    GlowLight->SetLightColor(FLinearColor(1.0f, 0.35f, 0.05f));
    FlashIntensity = 1.2f;
    FlashTimer     = PulseFlashDuration * 2.0f;   // longer alert flash
}

// ---------------------------------------------------------------------------
// FadeOut
// ---------------------------------------------------------------------------

void ANeuronNodeActor::FadeOut()
{
    bFading   = true;
    FadeTimer = FadeOutDuration;
    OnPruneStart();
}

// ---------------------------------------------------------------------------
// UpdateVisuals (per-Tick)
// ---------------------------------------------------------------------------

void ANeuronNodeActor::UpdateVisuals(float DeltaTime)
{
    // --- Pulse flash decay ---
    if (FlashTimer > 0.0f)
    {
        FlashTimer -= DeltaTime;
        const float T = FMath::Max(FlashTimer / PulseFlashDuration, 0.0f);

        // Lerp light radius from peak back toward resting
        GlowLight->SetAttenuationRadius(
            FMath::Lerp(RestingLightRadius,
                        FMath::Lerp(RestingLightRadius, PeakLightRadius, FlashIntensity),
                        T));
        GlowLight->SetLightColor(BaseColor);

        if (SphereMID)
        {
            SphereMID->SetScalarParameterValue(TEXT("EmissiveIntensity"),
                                               FMath::Lerp(0.3f, 3.0f * FlashIntensity, T));
            SphereMID->SetVectorParameterValue(TEXT("BaseColor"), BaseColor);
        }
    }

    // --- Fade-out ---
    if (bFading)
    {
        FadeTimer -= DeltaTime;
        const float Alpha = FMath::Max(FadeTimer / FadeOutDuration, 0.0f);

        if (SphereMID)
        {
            SphereMID->SetScalarParameterValue(TEXT("Opacity"), Alpha);
        }
        GlowLight->SetIntensity(500.0f * Alpha);

        if (FadeTimer <= 0.0f)
        {
            Destroy();
        }
    }
}

// ---------------------------------------------------------------------------
// CommunityToColor  (static helper)
// ---------------------------------------------------------------------------

FLinearColor ANeuronNodeActor::CommunityToColor(int32 CID)
{
    // Distribute communities around the HSV hue wheel (golden ratio step)
    // so adjacent integer IDs get perceptually distinct colours.
    constexpr float GoldenRatioConj = 0.6180339887f;
    const float Hue = FMath::Fmod(CID * GoldenRatioConj, 1.0f);
    return FLinearColor::MakeFromHSV8(
        static_cast<uint8>(Hue * 255),
        200,    // Saturation
        230);   // Value
}
