#include "CerebrumLink.h"
#include "Engine/Engine.h"

UCerebrumLink::UCerebrumLink() 
{ 
    PrimaryComponentTick.bCanEverTick = false; 
}

void UCerebrumLink::ConnectToBrain(FString URL) 
{
    UE_LOG(LogTemp, Log, TEXT("CerebrumLink: Attempting connection to %s"), *URL);
}

void UCerebrumLink::SendToBrain(FString Message)
{
    UE_LOG(LogTemp, Log, TEXT("CerebrumLink: Sending: %s"), *Message);
}
