#pragma once

#include "CoreMinimal.h"
#include "Blueprint/UserWidget.h"
#include "Components/ProgressBar.h"
#include "Components/TextBlock.h"
#include "CerebrumHUDOverlay.generated.h"

/**
 * UCerebrumHUDOverlay
 *
 * UMG base class for the metabolic blood-panel HUD.
 * Create WBP_CerebrumHUDOverlay extending this class. The Blueprint must
 * contain ProgressBars named exactly:
 *   ReinforcementBar   (Dopamine    — green)
 *   ArousalBar         (NE          — orange)
 *   NoveltyBar         (ACh         — cyan)
 *   CohesionBar        (Oxytocin    — purple)
 *   PersistenceBar     (Vasopressin — red)
 * Optional TextBlocks:
 *   ConnectionStatusText
 *   EventCounterText
 *   NodeCountText
 *
 * Wire BP_CerebrumBrain.OnMetabolicUpdate → this.UpdateMetabolics.
 */
UCLASS(Abstract, Blueprintable, BlueprintType)
class CEREBRUMVISUALIZER_API UCerebrumHUDOverlay : public UUserWidget
{
    GENERATED_BODY()

public:
    UPROPERTY(meta=(BindWidget))         UProgressBar* ReinforcementBar;
    UPROPERTY(meta=(BindWidget))         UProgressBar* ArousalBar;
    UPROPERTY(meta=(BindWidget))         UProgressBar* NoveltyBar;
    UPROPERTY(meta=(BindWidget))         UProgressBar* CohesionBar;
    UPROPERTY(meta=(BindWidget))         UProgressBar* PersistenceBar;
    UPROPERTY(meta=(BindWidgetOptional)) UTextBlock*   ConnectionStatusText;
    UPROPERTY(meta=(BindWidgetOptional)) UTextBlock*   EventCounterText;
    UPROPERTY(meta=(BindWidgetOptional)) UTextBlock*   NodeCountText;

    UFUNCTION(BlueprintCallable, Category="Cerebrum|HUD")
    void UpdateMetabolics(float Reinforcement, float Arousal,
                          float Novelty, float Cohesion, float Persistence);

    UFUNCTION(BlueprintCallable, Category="Cerebrum|HUD")
    void SetConnectionStatus(bool bConnected, const FString& URL);

    UFUNCTION(BlueprintCallable, Category="Cerebrum|HUD")
    void IncrementEventCounter(const FString& EventType);

    UFUNCTION(BlueprintCallable, Category="Cerebrum|HUD")
    void UpdateNodeCount(int32 NodeCount, int32 SynapseCount);

private:
    int32 TotalEventCount = 0;
};
