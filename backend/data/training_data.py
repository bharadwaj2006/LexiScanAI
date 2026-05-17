"""
Annotated legal contract training data for SpaCy NER fine-tuning.

Format: List of (text, {"entities": [(start, end, label), ...]})

Labels: LEGAL_DATE | LEGAL_AMOUNT | LEGAL_PARTY | TERMINATION_CLAUSE

Note: Character offsets are byte-exact; verify with:
    text[start:end] == label_text
"""

TRAINING_DATA = [
    (
        "This Agreement is entered into as of January 15, 2024 by and between "
        "Acme Corporation and GlobalTech Industries LLC.",
        {
            "entities": [
                (38, 54, "LEGAL_DATE"),       # January 15, 2024
                (72, 88, "LEGAL_PARTY"),       # Acme Corporation
                (93, 120, "LEGAL_PARTY"),      # GlobalTech Industries LLC
            ]
        },
    ),
    (
        "The total fee shall be $2,500,000 payable in four equal installments of $625,000.",
        {
            "entities": [
                (22, 33, "LEGAL_AMOUNT"),      # $2,500,000
                (72, 80, "LEGAL_AMOUNT"),      # $625,000
            ]
        },
    ),
    (
        "Either party may terminate this Agreement upon ninety (90) days prior written notice.",
        {
            "entities": [
                (0, 85, "TERMINATION_CLAUSE"),
            ]
        },
    ),
    (
        "Effective Date: March 1, 2024. Expiry Date: February 28, 2026.",
        {
            "entities": [
                (16, 29, "LEGAL_DATE"),   # March 1, 2024
                (43, 61, "LEGAL_DATE"),   # February 28, 2026
            ]
        },
    ),
    (
        "Client shall pay USD 150,000 per annum to Sterling Law Partners LLP.",
        {
            "entities": [
                (21, 28, "LEGAL_AMOUNT"),      # 150,000  (USD prefix included via ruler)
                (48, 69, "LEGAL_PARTY"),       # Sterling Law Partners LLP
            ]
        },
    ),
    (
        "This Agreement shall terminate immediately if Meridian Capital Group breaches any material term.",
        {
            "entities": [
                (0, 94, "TERMINATION_CLAUSE"),
                (45, 68, "LEGAL_PARTY"),
            ]
        },
    ),
    (
        "The contract period runs from 01/01/2023 to 31/12/2025.",
        {
            "entities": [
                (30, 40, "LEGAL_DATE"),   # 01/01/2023
                (44, 54, "LEGAL_DATE"),   # 31/12/2025
            ]
        },
    ),
    (
        "Vertex Analytics Inc. agrees to pay GBP 500,000 to Horizon Consulting Ltd. by 2024-06-30.",
        {
            "entities": [
                (0, 21, "LEGAL_PARTY"),         # Vertex Analytics Inc.
                (35, 46, "LEGAL_AMOUNT"),        # GBP 500,000
                (50, 74, "LEGAL_PARTY"),         # Horizon Consulting Ltd.
                (79, 89, "LEGAL_DATE"),          # 2024-06-30
            ]
        },
    ),
    (
        "Termination for cause requires a 30-day notice period under Section 12.3.",
        {
            "entities": [(0, 71, "TERMINATION_CLAUSE")]
        },
    ),
    (
        "The Effective Date of this NDA is the 15th day of September, 2023.",
        {
            "entities": [(38, 66, "LEGAL_DATE")]
        },
    ),
    (
        "Nexus Financial Services Corporation shall indemnify the Client for losses up to $10,000,000.",
        {
            "entities": [
                (0, 38, "LEGAL_PARTY"),
                (81, 93, "LEGAL_AMOUNT"),
            ]
        },
    ),
    (
        "Upon expiration or early termination of this Agreement on December 31, 2025, "
        "all licenses shall immediately cease.",
        {
            "entities": [
                (0, 110, "TERMINATION_CLAUSE"),
                (60, 76, "LEGAL_DATE"),
            ]
        },
    ),
    (
        "The parties agree that Blue Ridge Capital Partners LLC owes $3,750,000 to "
        "Pacific Investment Group by April 30, 2024.",
        {
            "entities": [
                (23, 53, "LEGAL_PARTY"),      # Blue Ridge Capital Partners LLC
                (59, 71, "LEGAL_AMOUNT"),     # $3,750,000
                (75, 100, "LEGAL_PARTY"),     # Pacific Investment Group
                (104, 118, "LEGAL_DATE"),     # April 30, 2024
            ]
        },
    ),
    (
        "Service Provider may terminate with immediate effect if Client fails to pay "
        "within fifteen (15) days of the invoice date.",
        {
            "entities": [(0, 123, "TERMINATION_CLAUSE")]
        },
    ),
    (
        "This Lease Agreement between Silverton Properties Ltd. and Quantum Retail Corp. "
        "commences on July 1, 2023.",
        {
            "entities": [
                (29, 53, "LEGAL_PARTY"),
                (58, 77, "LEGAL_PARTY"),
                (92, 106, "LEGAL_DATE"),
            ]
        },
    ),
    (
        "The initial payment of $850,000 is due on 15/03/2024, with the final installment "
        "of $1,200,000 on 30/09/2024.",
        {
            "entities": [
                (22, 30, "LEGAL_AMOUNT"),
                (42, 52, "LEGAL_DATE"),
                (82, 94, "LEGAL_AMOUNT"),
                (98, 108, "LEGAL_DATE"),
            ]
        },
    ),
    (
        "Either party may cancel this Agreement without cause by providing sixty (60) days "
        "advance written notice to the other party.",
        {
            "entities": [(0, 123, "TERMINATION_CLAUSE")]
        },
    ),
    (
        "Ironclad Legal Solutions Inc. entered a retainer of USD 75,000 per month "
        "commencing January 1, 2024.",
        {
            "entities": [
                (0, 29, "LEGAL_PARTY"),
                (51, 61, "LEGAL_AMOUNT"),
                (83, 98, "LEGAL_DATE"),
            ]
        },
    ),
    (
        "The Agreement expires on the last day of the Term, being December 31, 2027, "
        "unless renewed in writing.",
        {
            "entities": [(60, 76, "LEGAL_DATE")]
        },
    ),
    (
        "Upon any termination event, Broadmark Asset Management LLC shall deliver "
        "all client data within five (5) business days.",
        {
            "entities": [
                (0, 122, "TERMINATION_CLAUSE"),
                (28, 57, "LEGAL_PARTY"),
            ]
        },
    ),
]
