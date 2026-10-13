
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

CATEGORIES = ["Billing", "Technical", "Delivery", "Account"]

# 25 different complaint types per category
COMPLAINTS = {
    "Billing": [
        "I was charged twice for one purchase",
        "my refund has not arrived after ten days",
        "my invoice contains an incorrect amount",
        "a subscription payment was taken after cancellation",
        "my bank statement shows an unknown service charge",
        "I paid the bill but the system still says unpaid",
        "the discount was not applied to my order",
        "I need a refund for a cancelled purchase",
        "the monthly fee is higher than expected",
        "I received a duplicate payment receipt",
        "my card was charged for an order I never placed",
        "the final bill includes an unexplained fee",
        "my refund was approved but the money is missing",
        "I cannot find the receipt for my recent payment",
        "the same subscription fee appears multiple times",
        "my promotional coupon did not reduce the price",
        "I was charged after my free trial ended unexpectedly",
        "the payment amount differs from the checkout amount",
        "my payment failed but money was deducted",
        "I need clarification about a tax on my invoice",
        "the company has not processed my refund request",
        "my account shows an outstanding balance after payment",
        "I was billed for a service I did not use",
        "the installment amount is incorrect",
        "I need to update the billing details on my invoice"
    ],

    "Technical": [
        "the mobile application crashes when I open it",
        "the website shows a server error",
        "the app freezes every time I upload a file",
        "the payment page displays a technical error",
        "images are not loading on the website",
        "the application keeps closing unexpectedly",
        "the search feature does not return any results",
        "the latest update made the application very slow",
        "I cannot upload documents because of an error",
        "the website displays a blank screen",
        "the application crashes when I try to log in",
        "the download button does not respond",
        "the website takes too long to load",
        "I receive an error when saving my changes",
        "the mobile app stops working after an update",
        "the video player shows a playback error",
        "the page layout breaks on my computer",
        "the application cannot connect to the server",
        "notifications are not appearing in the app",
        "the file upload gets stuck halfway",
        "the website logs me out while I am browsing",
        "the application displays incorrect search results",
        "the settings page does not open",
        "the software fails when processing large files",
        "the website is not responding to clicks"
    ],

    "Delivery": [
        "my package has not arrived on the promised date",
        "the tracking page has not updated for a week",
        "my order was delivered to the wrong address",
        "the parcel arrived with damaged packaging",
        "one item is missing from my delivered package",
        "the courier keeps postponing the delivery",
        "I received a notification but no package arrived",
        "my shipment is stuck at the sorting facility",
        "the delivery person could not find my address",
        "my order was marked delivered but I did not receive it",
        "the tracking number does not show any information",
        "my package was returned without my permission",
        "the courier delivered my order to another person",
        "my shipment has been delayed several times",
        "the parcel arrived later than the estimated date",
        "I received the wrong product in my delivery",
        "the package was left outside in the rain",
        "the delivery address was changed without my request",
        "my express shipment has not arrived on time",
        "the courier has not contacted me about delivery",
        "part of my order was missing from the parcel",
        "the package status has been stuck in transit",
        "I was charged for delivery but the order was not shipped",
        "the delivery agent marked an unsuccessful attempt without visiting",
        "I need to reschedule my package delivery"
    ],

    "Account": [
        "I cannot reset my account password",
        "my account has been locked unexpectedly",
        "I cannot log in with my correct credentials",
        "I need to change the email linked to my profile",
        "the verification code for account access never arrives",
        "I want to update my profile information",
        "my account login keeps asking for verification",
        "I forgot my password and cannot access my profile",
        "my account was suspended and I need access restored",
        "I cannot change my registered phone number",
        "my username is not being recognized",
        "I cannot complete the account registration process",
        "my profile information is not being saved",
        "I need to deactivate my account",
        "I cannot verify my email address",
        "my account recovery link has expired",
        "I am unable to enable two-factor authentication",
        "my account is showing someone else's profile details",
        "I need to change my account username",
        "I cannot remove an old phone number from my profile",
        "my login session expires immediately",
        "I received an unexpected account security alert",
        "my account remains restricted after identity verification",
        "I cannot link my account to a new device",
        "I need help recovering access to my account"
    ]
}

# Ten different sentence patterns for each complaint type.
# This creates 25 x 10 = 250 samples per category.
TEMPLATES = [
    "{issue}.",
    "Please help me resolve this: {issue}.",
    "I'm contacting support because {issue}.",
    "I'd like to report that {issue}.",
    "Could you investigate why {issue}?",
    "I need assistance because {issue}.",
    "This situation is frustrating because {issue}.",
    "I noticed that {issue}.",
    "Can someone look into this? {issue}.",
    "Please advise me because {issue}."
]


def generate_dataset():
    rows = []

    for category, issues in COMPLAINTS.items():
        for issue in issues:
            for template in TEMPLATES:
                complaint = template.format(issue=issue)

                # Capitalize the beginning of each complaint.
                complaint = complaint[0].upper() + complaint[1:]

                rows.append({
                    "text": complaint,
                    "label": category
                })

    df = pd.DataFrame(rows)

    # Remove any exact duplicate rows, if present.
    df = df.drop_duplicates(subset=["text", "label"])
    df = df.reset_index(drop=True)

    return df


def prepare_data():
    data_dir = Path("data")
    results_dir = Path("results")

    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    df = generate_dataset()

    # Save the complete dataset.
    df.to_csv(data_dir / "complaints.csv", index=False)

    # Split data while preserving category proportions.
    train_df, test_df = train_test_split(
        df,
        test_size=0.25,
        random_state=42,
        stratify=df["label"]
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df.to_csv(results_dir / "train_data.csv", index=False)
    test_df.to_csv(results_dir / "test_data.csv", index=False)

    print("\nDataset generated successfully!")
    print(f"Total complaints: {len(df)}")
    print(f"Training complaints: {len(train_df)}")
    print(f"Testing complaints: {len(test_df)}")
    print("\nComplaints per category:")
    print(df["label"].value_counts().sort_index())

    return train_df, test_df


if __name__ == "__main__":
    prepare_data()
