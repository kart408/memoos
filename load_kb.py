from memoos_core.knowledge_base import KnowledgeBase

kb = KnowledgeBase(client_id="demo_college")

documents = [
    ("The admission deadline for the AI/ML Engineering program is August 15th. "
     "Applications require academic transcripts, a statement of purpose, and a portfolio.",
     "admissions_deadline"),

    ("The annual tuition fee for the AI/ML Engineering program is 1.8 lakh rupees. "
     "Hostel accommodation costs an additional 90,000 rupees per year, and is optional.",
     "fees"),

    ("Merit-based scholarships covering up to 50% of tuition are available for students "
     "with strong academic records. Need-based financial aid is also available. "
     "Scholarship recipients must maintain a minimum CGPA of 8.0 each year.",
     "scholarships"),

    ("Last year's AI/ML Engineering batch had a 92% placement rate, with average packages "
     "around 9 lakh rupees per year, and top offers up to 28 lakh rupees per year.",
     "placements"),

    ("On-campus hostel accommodation is available for both men and women, with priority "
     "given to students from outside Bengaluru.",
     "hostel"),

    ("Applicants must clear the institute's entrance exam, which tests mathematics, "
     "logical reasoning, and basic programming concepts.",
     "entrance_exam"),

    ("The first-year curriculum covers Python and C++, along with foundational courses "
     "in data structures, discrete mathematics, and linear algebra.",
     "curriculum"),

    ("Students typically complete a mandatory internship in their third year, coordinated "
     "through the placement cell, with options in AI research labs, startups, or industry partners.",
     "internships"),

    ("Undergraduates can join faculty-led research labs starting in their second year, "
     "with active groups in computer vision, NLP, and robotics.",
     "research"),

    ("The admissions office can be reached at admissions@bitech.edu.in, or by phone "
     "between 9 AM and 5 PM on weekdays.",
     "contact"),
]

total_chunks = 0
for text, name in documents:
    count = kb.add_document(text, doc_name=name)
    total_chunks += count
    print(f"Added '{name}' ({count} chunk(s))")

print(f"\nDone. {total_chunks} total chunks in the knowledge base.")