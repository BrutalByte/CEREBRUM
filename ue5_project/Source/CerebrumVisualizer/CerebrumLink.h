#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "CerebrumLink.generated.h"

// Define the delegate for receiving neural events
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnNeuralEventReceived, FString, JsonData);

UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class CEREBRUMVISUALIZER_API UCerebrumLink : public UActorComponent
{
    GENERATED_BODY()

public:
    UCerebrumLink();

    UPROPERTY(BlueprintAssignable, Category = "Cerebrum")
    FOnNeuralEventReceived OnNeuralEvent;

    UFUNCTION(BlueprintCallable, Category = "Cerebrum")
    void ConnectToBrain(FString URL);

    UFUNCTION(BlueprintCallable, Category = "Cerebrum")
    void SendToBrain(FString Message);
};
