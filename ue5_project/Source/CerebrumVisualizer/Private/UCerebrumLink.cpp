// UCerebrumLink.cpp
#include "UCerebrumLink.h"
#include "WebSocketsModule.h"
#include "Json.h"
#include "JsonUtilities.h"

UUCerebrumLink::UUCerebrumLink() {
    PrimaryComponentTick.bCanEverTick = false;
}

void UUCerebrumLink::BeginPlay() {
    Super::BeginPlay();

    if (!FModuleManager::Get().IsModuleLoaded("WebSockets")) {
        FModuleManager::Get().LoadModule("WebSockets");
    }

    WebSocket = FWebSocketsModule::Get().CreateWebSocket("ws://localhost:8765");
    
    WebSocket->OnMessage().AddLambda([this](const FString& Message) {
        OnMessage(Message);
    });

    WebSocket->Connect();
}

void UUCerebrumLink::EndPlay(const EEndPlayReason::Type EndPlayReason) {
    if (WebSocket.IsValid() && WebSocket->IsConnected()) {
        WebSocket->Close();
    }
    Super::EndPlay(EndPlayReason);
}

void UUCerebrumLink::OnMessage(const FString& Message) {
    TSharedPtr<FJsonObject> JsonObject;
    TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Message);

    if (FJsonSerializer::Deserialize(Reader, JsonObject)) {
        FNeuralEvent Event;
        if (FJsonObjectConverter::JsonObjectToUStruct(JsonObject.ToSharedRef(), &Event)) {
            AsyncTask(ENamedThreads::GameThread, [this, Event]() {
                OnNeuralEvent.Broadcast(Event);
            });
        }
    }
}
