#include "CerebrumHUDOverlay.h"

void UCerebrumHUDOverlay::UpdateMetabolics(
    float Reinforcement, float Arousal,
    float Novelty, float Cohesion, float Persistence)
{
    if (ReinforcementBar) ReinforcementBar->SetPercent(FMath::Clamp(Reinforcement, 0.f, 1.f));
    if (ArousalBar)       ArousalBar      ->SetPercent(FMath::Clamp(Arousal,       0.f, 1.f));
    if (NoveltyBar)       NoveltyBar      ->SetPercent(FMath::Clamp(Novelty,       0.f, 1.f));
    if (CohesionBar)      CohesionBar     ->SetPercent(FMath::Clamp(Cohesion,      0.f, 1.f));
    if (PersistenceBar)   PersistenceBar  ->SetPercent(FMath::Clamp(Persistence,   0.f, 1.f));
}

void UCerebrumHUDOverlay::SetConnectionStatus(bool bConnected, const FString& URL)
{
    if (!ConnectionStatusText) return;

    ConnectionStatusText->SetText(FText::FromString(
        bConnected
            ? FString::Printf(TEXT("● LIVE  %s"), *URL)
            : TEXT("○  DISCONNECTED")));

    // #3fb950 green when connected, #f85149 red when not
    FLinearColor Color = bConnected
        ? FLinearColor(0.243f, 0.714f, 0.314f, 1.f)
        : FLinearColor(0.973f, 0.318f, 0.286f, 1.f);
    ConnectionStatusText->SetColorAndOpacity(FSlateColor(Color));
}

void UCerebrumHUDOverlay::IncrementEventCounter(const FString& EventType)
{
    TotalEventCount++;
    if (EventCounterText)
    {
        EventCounterText->SetText(FText::FromString(
            FString::Printf(TEXT("Events: %d  |  %s"), TotalEventCount, *EventType)));
    }
}

void UCerebrumHUDOverlay::UpdateNodeCount(int32 NodeCount, int32 SynapseCount)
{
    if (NodeCountText)
    {
        NodeCountText->SetText(FText::FromString(
            FString::Printf(TEXT("%d nodes  ·  %d synapses"), NodeCount, SynapseCount)));
    }
}
