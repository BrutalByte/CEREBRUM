#pragma once

#include "CoreMinimal.h"
#include "Blueprint/UserWidget.h"
#include "Components/EditableTextBox.h"
#include "Components/Button.h"
#include "Components/ScrollBox.h"
#include "Components/TextBlock.h"
#include "HttpModule.h"
#include "Interfaces/IHttpRequest.h"
#include "Interfaces/IHttpResponse.h"
#include "CerebrumQueryWidget.generated.h"

/**
 * UCerebrumQueryWidget
 *
 * UMG base class for the in-world query panel.
 * Create WBP_CerebrumQueryPanel extending this class. The Blueprint must
 * contain widgets with these exact names:
 *   QueryInput    (UEditableTextBox)
 *   SubmitButton  (UButton)
 *   ResultsScroll (UScrollBox)   — optional
 *   StatusLabel   (UTextBlock)   — optional
 */
UCLASS(Abstract, Blueprintable, BlueprintType)
class CEREBRUMVISUALIZER_API UCerebrumQueryWidget : public UUserWidget
{
    GENERATED_BODY()

public:
    UPROPERTY(meta=(BindWidget))
    UEditableTextBox* QueryInput;

    UPROPERTY(meta=(BindWidget))
    UButton* SubmitButton;

    UPROPERTY(meta=(BindWidgetOptional))
    UScrollBox* ResultsScroll;

    UPROPERTY(meta=(BindWidgetOptional))
    UTextBlock* StatusLabel;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cerebrum|Query")
    FString RESTBaseURL = TEXT("http://localhost:8200");

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cerebrum|Query")
    FString AuthToken;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cerebrum|Query")
    int32 MaxHops = 3;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cerebrum|Query")
    int32 BeamWidth = 5;

    UFUNCTION(BlueprintCallable, Category="Cerebrum|Query")
    void SubmitQuery(const FString& QueryText);

    UFUNCTION(BlueprintImplementableEvent, Category="Cerebrum|Query")
    void OnQueryComplete(const FString& ResultJson, const TArray<FString>& Answers);

    UFUNCTION(BlueprintImplementableEvent, Category="Cerebrum|Query")
    void OnQueryError(const FString& ErrorMessage);

protected:
    virtual void NativeConstruct() override;

private:
    UFUNCTION()
    void OnSubmitClicked();

    void OnHTTPResponse(FHttpRequestPtr Req, FHttpResponsePtr Resp, bool bConnectedSuccessfully);
};
