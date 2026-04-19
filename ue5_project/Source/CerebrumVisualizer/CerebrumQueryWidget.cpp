#include "CerebrumQueryWidget.h"
#include "Json.h"
#include "JsonUtilities.h"

void UCerebrumQueryWidget::NativeConstruct()
{
    Super::NativeConstruct();
    if (SubmitButton)
    {
        SubmitButton->OnClicked.AddDynamic(this, &UCerebrumQueryWidget::OnSubmitClicked);
    }
}

void UCerebrumQueryWidget::OnSubmitClicked()
{
    if (QueryInput)
    {
        SubmitQuery(QueryInput->GetText().ToString());
    }
}

void UCerebrumQueryWidget::SubmitQuery(const FString& QueryText)
{
    if (QueryText.IsEmpty()) return;

    if (StatusLabel)
    {
        StatusLabel->SetText(FText::FromString(TEXT("Querying...")));
    }

    FString SafeQuery = QueryText.Replace(TEXT("\""), TEXT("\\\""));
    FString Body = FString::Printf(
        TEXT("{\"query\":\"%s\",\"max_hop\":%d,\"beam_width\":%d}"),
        *SafeQuery, MaxHops, BeamWidth);

    TSharedRef<IHttpRequest, ESPMode::ThreadSafe> Req = FHttpModule::Get().CreateRequest();
    Req->SetURL(RESTBaseURL + TEXT("/v1/query"));
    Req->SetVerb(TEXT("POST"));
    Req->SetHeader(TEXT("Content-Type"), TEXT("application/json"));
    if (!AuthToken.IsEmpty())
    {
        Req->SetHeader(TEXT("Authorization"), TEXT("Bearer ") + AuthToken);
    }
    Req->SetContentAsString(Body);
    Req->OnProcessRequestComplete().BindUObject(this, &UCerebrumQueryWidget::OnHTTPResponse);
    Req->ProcessRequest();
}

void UCerebrumQueryWidget::OnHTTPResponse(
    FHttpRequestPtr Req, FHttpResponsePtr Resp, bool bConnectedSuccessfully)
{
    if (!bConnectedSuccessfully || !Resp)
    {
        OnQueryError(TEXT("Network error — is the CEREBRUM server running?"));
        if (StatusLabel) StatusLabel->SetText(FText::FromString(TEXT("Error")));
        return;
    }

    if (Resp->GetResponseCode() != 200)
    {
        OnQueryError(FString::Printf(TEXT("HTTP %d"), Resp->GetResponseCode()));
        if (StatusLabel)
        {
            StatusLabel->SetText(FText::FromString(
                FString::Printf(TEXT("Error %d"), Resp->GetResponseCode())));
        }
        return;
    }

    TSharedPtr<FJsonObject> JsonObj;
    TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Resp->GetContentAsString());
    TArray<FString> Answers;

    if (FJsonSerializer::Deserialize(Reader, JsonObj) && JsonObj.IsValid())
    {
        const TArray<TSharedPtr<FJsonValue>>* ArrPtr;
        if (JsonObj->TryGetArrayField(TEXT("answers"), ArrPtr))
        {
            for (const TSharedPtr<FJsonValue>& Val : *ArrPtr)
            {
                FString AnswerStr;
                if (Val->TryGetString(AnswerStr))
                {
                    Answers.Add(AnswerStr);
                }
                else
                {
                    const TSharedPtr<FJsonObject>* ObjPtr;
                    if (Val->TryGetObject(ObjPtr) && ObjPtr && (*ObjPtr).IsValid())
                    {
                        FString Entity;
                        if ((*ObjPtr)->TryGetStringField(TEXT("entity"), Entity))
                        {
                            Answers.Add(Entity);
                        }
                    }
                }
            }
        }
    }

    OnQueryComplete(Resp->GetContentAsString(), Answers);

    if (StatusLabel)
    {
        StatusLabel->SetText(FText::FromString(
            FString::Printf(TEXT("%d answer%s"),
                Answers.Num(), Answers.Num() == 1 ? TEXT("") : TEXT("s"))));
    }
}
