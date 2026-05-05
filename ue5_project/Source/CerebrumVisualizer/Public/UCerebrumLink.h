// UCerebrumLink.h
#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "IWebSocket.h"
#include "UCerebrumLink.generated.h"

USTRUCT(BlueprintType)
struct FNeuralEvent {
    GENERATED_BODY()
    UPROPERTY(BlueprintReadOnly) FString EventType;
    UPROPERTY(BlueprintReadOnly) FString SourceNode;
    UPROPERTY(BlueprintReadOnly) FString TargetNode;
    UPROPERTY(BlueprintReadOnly) float Weight;
};

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnNeuralEventReceived, const FNeuralEvent&, Event);

UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class CEREBRUM_VISUALIZER_API UUCerebrumLink : public UActorComponent
{
    GENERATED_BODY()

public:
    UUCerebrumLink();

    UPROPERTY(BlueprintAssignable, Category = "Cerebrum")
    FOnNeuralEventReceived OnNeuralEvent;

protected:
    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

private:
    TSharedPtr<IWebSocket> WebSocket;
    void OnMessage(const FString& Message);
};
