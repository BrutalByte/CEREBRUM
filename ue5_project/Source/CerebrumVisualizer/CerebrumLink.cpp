#include "CerebrumLink.h"
#include "WebsocketFunctionLibrary.h"
#include "Json.h"
#include "JsonUtilities.h"
#include "Engine/Engine.h"

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

UCerebrumLink::UCerebrumLink()
{
    PrimaryComponentTick.bCanEverTick = false;
    Socket = nullptr;
}

// ---------------------------------------------------------------------------
// Connection management
// ---------------------------------------------------------------------------

void UCerebrumLink::ConnectToBrain(FString URL)
{
    if (Socket && Socket->IsConnected())
    {
        UE_LOG(LogTemp, Warning,
               TEXT("CerebrumLink: Already connected. Call Disconnect() first."));
        return;
    }

    UE_LOG(LogTemp, Log, TEXT("CerebrumLink: Connecting to %s"), *URL);

    Socket = UWebSocketFunctionLibrary::CreateWebSocket(URL, TEXT("ws"));
    if (!Socket)
    {
        UE_LOG(LogTemp, Error, TEXT("CerebrumLink: Failed to create WebSocket for %s"), *URL);
        return;
    }

    // Bind all handlers before connecting
    Socket->OnWebSocketConnected.AddDynamic(this, &UCerebrumLink::HandleConnected);
    Socket->OnWebSocketConnectionError.AddDynamic(this, &UCerebrumLink::HandleConnectionError);
    Socket->OnWebSocketClosed.AddDynamic(this, &UCerebrumLink::HandleClosed);
    Socket->OnWebSocketMessageReceived.AddDynamic(this, &UCerebrumLink::HandleMessage);

    Socket->Connect();
}

void UCerebrumLink::Disconnect()
{
    if (Socket)
    {
        Socket->Close(1000, TEXT("client_disconnect"));
    }
}

bool UCerebrumLink::IsConnected() const
{
    return Socket != nullptr && Socket->IsConnected();
}

void UCerebrumLink::SendToBrain(FString Message)
{
    if (!IsConnected())
    {
        UE_LOG(LogTemp, Warning,
               TEXT("CerebrumLink: SendToBrain called while not connected."));
        return;
    }
    Socket->SendMessage(Message);
}

// ---------------------------------------------------------------------------
// WebSocket internal handlers
// ---------------------------------------------------------------------------

void UCerebrumLink::HandleConnected()
{
    UE_LOG(LogTemp, Log, TEXT("CerebrumLink: Connected to CEREBRUM telemetry bridge."));
}

void UCerebrumLink::HandleConnectionError(const FString& Error)
{
    UE_LOG(LogTemp, Error, TEXT("CerebrumLink: Connection error: %s"), *Error);
    Socket = nullptr;
}

void UCerebrumLink::HandleClosed(int32 StatusCode, const FString& Reason, bool bWasClean)
{
    UE_LOG(LogTemp, Log,
           TEXT("CerebrumLink: Connection closed (code=%d, reason=%s, clean=%s)"),
           StatusCode, *Reason, bWasClean ? TEXT("true") : TEXT("false"));
    Socket = nullptr;
}

void UCerebrumLink::HandleMessage(const FString& JsonData)
{
    // 1. Fire generic delegate — any Blueprint already bound to OnNeuralEvent
    //    continues to work unchanged.
    OnNeuralEvent.Broadcast(JsonData);

    // 2. Parse JSON envelope
    TSharedPtr<FJsonObject> RootObject;
    TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(JsonData);
    if (!FJsonSerializer::Deserialize(Reader, RootObject) || !RootObject.IsValid())
    {
        UE_LOG(LogTemp, Warning,
               TEXT("CerebrumLink: Failed to parse neural event JSON: %s"), *JsonData);
        return;
    }

    // 3. Extract event_type and payload
    FString EventType;
    if (!RootObject->TryGetStringField(TEXT("event_type"), EventType))
    {
        // Silently ignore — might be a non-event control message
        return;
    }

    const TSharedPtr<FJsonObject>* PayloadPtr = nullptr;
    if (!RootObject->TryGetObjectField(TEXT("payload"), PayloadPtr) || !PayloadPtr)
    {
        UE_LOG(LogTemp, Warning,
               TEXT("CerebrumLink: Neural event '%s' has no payload field."), *EventType);
        return;
    }

    // 4. Dispatch typed delegate
    DispatchTypedEvent(EventType, *PayloadPtr);
}

// ---------------------------------------------------------------------------
// Typed event dispatch
// ---------------------------------------------------------------------------

void UCerebrumLink::DispatchTypedEvent(const FString& EventType,
                                       const TSharedPtr<FJsonObject>& Payload)
{
    if (EventType == TEXT("SYNAPTIC_PULSE"))
    {
        FString SourceNode, TargetNode, Relation;
        double  Weight    = 0.0;
        int32   HopCount  = 0;
        bool    bWormhole = false;

        Payload->TryGetStringField(TEXT("source_node"), SourceNode);
        Payload->TryGetStringField(TEXT("target_node"), TargetNode);
        Payload->TryGetStringField(TEXT("relation"),    Relation);
        Payload->TryGetNumberField(TEXT("weight"),      Weight);
        Payload->TryGetBoolField  (TEXT("is_wormhole"), bWormhole);

        // hop_count is stored as a number in JSON
        double HopDouble = 0.0;
        if (Payload->TryGetNumberField(TEXT("hop_count"), HopDouble))
        {
            HopCount = static_cast<int32>(HopDouble);
        }

        OnSynapticPulse.Broadcast(
            SourceNode, TargetNode, Relation,
            static_cast<float>(Weight), HopCount, bWormhole
        );
    }
    else if (EventType == TEXT("NEUROGENESIS"))
    {
        FString NodeId, Label, NodeType;
        Payload->TryGetStringField(TEXT("node_id"), NodeId);
        Payload->TryGetStringField(TEXT("label"),   Label);
        Payload->TryGetStringField(TEXT("type"),    NodeType);

        OnNeurogenesis.Broadcast(NodeId, Label, NodeType);
    }
    else if (EventType == TEXT("SYNAPTOGENESIS"))
    {
        // SYNAPTOGENESIS uses same payload layout as SYNAPTIC_PULSE (edge info)
        // Route through OnSynapticPulse with HopCount=0 as a creation marker,
        // or ignore if no Blueprint is bound — both are acceptable.
        FString SourceNode, TargetNode, Relation;
        double  Weight = 0.0;
        Payload->TryGetStringField(TEXT("source_node"), SourceNode);
        Payload->TryGetStringField(TEXT("target_node"), TargetNode);
        Payload->TryGetStringField(TEXT("relation"),    Relation);
        Payload->TryGetNumberField(TEXT("weight"),      Weight);
        // Distinguish from SYNAPTIC_PULSE in Blueprint via HopCount == -1
        OnSynapticPulse.Broadcast(SourceNode, TargetNode, Relation,
                                  static_cast<float>(Weight), -1, false);
    }
    else if (EventType == TEXT("SYNAPTIC_PRUNE"))
    {
        FString EdgeId, Reason;
        Payload->TryGetStringField(TEXT("edge_id"), EdgeId);
        Payload->TryGetStringField(TEXT("reason"),  Reason);

        OnSynapticPrune.Broadcast(EdgeId, Reason);
    }
    else if (EventType == TEXT("CORTICAL_GLOW"))
    {
        FString CommunityId;
        double  Intensity = 1.0;

        Payload->TryGetStringField(TEXT("community_id"), CommunityId);
        Payload->TryGetNumberField(TEXT("intensity"),    Intensity);

        OnCorticalGlow.Broadcast(CommunityId, static_cast<float>(Intensity));
    }
    else if (EventType == TEXT("DISSONANCE"))
    {
        FString SeedEntity;
        double  PathScore      = 0.0;
        double  ConsensusScore = 0.0;

        Payload->TryGetStringField(TEXT("seed_entity"),     SeedEntity);
        Payload->TryGetNumberField(TEXT("path_score"),      PathScore);
        Payload->TryGetNumberField(TEXT("consensus_score"), ConsensusScore);

        OnDissonance.Broadcast(
            SeedEntity,
            static_cast<float>(PathScore),
            static_cast<float>(ConsensusScore)
        );
    }
    else
    {
        // Unknown event type — already dispatched via generic OnNeuralEvent above.
        UE_LOG(LogTemp, VeryVerbose,
               TEXT("CerebrumLink: Unhandled event type '%s' (generic delegate fired)."),
               *EventType);
    }
}
