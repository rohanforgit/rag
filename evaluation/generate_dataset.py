import os
import sys
import json
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("generate_dataset")

# Add backend directory to sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
sys.path.append(backend_dir)

from app.services.embedder import EmbedderService
from app.services.retriever import RetrieverService
from app.services.llm import LLMService

# Define topics and questions
# We need at least 150 questions covering 24 topics
QUESTIONS_CATALOG = [
    # 1. Attendance
    {
        "question": "What is the minimum attendance percentage required in Sreenidhi University to appear for exams?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Attendance",
        "difficulty": "easy",
        "topic": "Attendance"
    },
    {
        "question": "Can I get condonation for attendance shortage if I have 70% attendance?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Attendance",
        "difficulty": "easy",
        "topic": "Attendance"
    },
    {
        "question": "What is the absolute minimum attendance percentage required even with condonation?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Attendance",
        "difficulty": "easy",
        "topic": "Attendance"
    },
    {
        "question": "Is condonation of attendance free, or is there a condonation fee?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Attendance",
        "difficulty": "medium",
        "topic": "Attendance"
    },
    {
        "question": "What happens if my attendance in a course is less than 65%?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Attendance",
        "difficulty": "medium",
        "topic": "Attendance"
    },
    {
        "question": "Do I have to register again for a course if I am detained due to low attendance?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Attendance",
        "difficulty": "medium",
        "topic": "Attendance"
    },
    {
        "question": "How is attendance recorded for students representing the college in sports events?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Attendance",
        "difficulty": "hard",
        "topic": "Attendance"
    },

    # 2. CIE
    {
        "question": "What is the weightage of Continuous Internal Evaluation (CIE) in the overall evaluation?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Evaluation",
        "difficulty": "easy",
        "topic": "CIE"
    },
    {
        "question": "How many mid-term exams are conducted in a semester for CIE?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Evaluation",
        "difficulty": "easy",
        "topic": "CIE"
    },
    {
        "question": "What are the components that make up the CIE marks for theory courses?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Evaluation",
        "difficulty": "medium",
        "topic": "CIE"
    },
    {
        "question": "Is there a minimum marks requirement in CIE to be eligible for Semester End Exams?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Evaluation",
        "difficulty": "medium",
        "topic": "CIE"
    },
    {
        "question": "How is the CIE evaluated for a laboratory course?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Evaluation",
        "difficulty": "medium",
        "topic": "CIE"
    },
    {
        "question": "What happens if a student misses a mid-term exam? Is there a makeup test?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Evaluation",
        "difficulty": "hard",
        "topic": "CIE"
    },
    {
        "question": "Are assignments and quizzes graded under CIE? If so, what is their contribution?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Evaluation",
        "difficulty": "hard",
        "topic": "CIE"
    },

    # 3. SEE
    {
        "question": "What is the weightage of the Semester End Examination (SEE) in a theory course?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Evaluation",
        "difficulty": "easy",
        "topic": "SEE"
    },
    {
        "question": "What is the minimum passing mark required in the Semester End Examination?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Evaluation",
        "difficulty": "easy",
        "topic": "SEE"
    },
    {
        "question": "How long is the duration of a theory Semester End Examination?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Evaluation",
        "difficulty": "medium",
        "topic": "SEE"
    },
    {
        "question": "What happens if a student is absent from the Semester End Examination?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Evaluation",
        "difficulty": "medium",
        "topic": "SEE"
    },
    {
        "question": "Are laboratory Semester End Examinations evaluated by external examiners?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Evaluation",
        "difficulty": "hard",
        "topic": "SEE"
    },
    {
        "question": "Is there any provision for a makeup examination if I miss SEE due to emergency?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Evaluation",
        "difficulty": "hard",
        "topic": "SEE"
    },

    # 4. Grades
    {
        "question": "What is the grade point associated with an 'O' letter grade?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Grading",
        "difficulty": "easy",
        "topic": "Grades"
    },
    {
        "question": "What is the minimum passing letter grade in any course?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Grading",
        "difficulty": "easy",
        "topic": "Grades"
    },
    {
        "question": "What does the 'F' grade mean, and what is its grade point?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Grading",
        "difficulty": "easy",
        "topic": "Grades"
    },
    {
        "question": "What letter grade is given to a student who is absent from the Semester End Examination?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Grading",
        "difficulty": "medium",
        "topic": "Grades"
    },
    {
        "question": "Can you explain the grading scale from 'O' to 'F' and their corresponding marks?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Grading",
        "difficulty": "medium",
        "topic": "Grades"
    },
    {
        "question": "Under what condition is an 'I' grade (incomplete) awarded?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Grading",
        "difficulty": "hard",
        "topic": "Grades"
    },

    # 5. SGPA
    {
        "question": "What is SGPA and how is it defined?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Grading",
        "difficulty": "easy",
        "topic": "SGPA"
    },
    {
        "question": "What is the mathematical formula used to calculate SGPA?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Grading",
        "difficulty": "medium",
        "topic": "SGPA"
    },
    {
        "question": "Does a course with an 'F' grade get included in the SGPA calculation?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Grading",
        "difficulty": "medium",
        "topic": "SGPA"
    },
    {
        "question": "Do non-credit courses (audit courses) affect the SGPA calculation?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Grading",
        "difficulty": "medium",
        "topic": "SGPA"
    },
    {
        "question": "If I pass a backlog course, how is my new SGPA calculated for that semester?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Grading",
        "difficulty": "hard",
        "topic": "SGPA"
    },

    # 6. CGPA
    {
        "question": "What is CGPA and how is it defined?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Grading",
        "difficulty": "easy",
        "topic": "CGPA"
    },
    {
        "question": "Write down the formula for calculating CGPA.",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Grading",
        "difficulty": "medium",
        "topic": "CGPA"
    },
    {
        "question": "How is SGPA different from CGPA?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Grading",
        "difficulty": "medium",
        "topic": "CGPA"
    },
    {
        "question": "What CGPA is required to be eligible for the award of First Class with Distinction?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Grading",
        "difficulty": "medium",
        "topic": "CGPA"
    },
    {
        "question": "What CGPA represents a simple pass class for graduation?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Grading",
        "difficulty": "hard",
        "topic": "CGPA"
    },
    {
        "question": "How do you convert CGPA to equivalent percentage under R26?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Grading",
        "difficulty": "hard",
        "topic": "CGPA"
    },

    # 7. Credit requirements
    {
        "question": "What is the total number of credits required to graduate B.Tech under R26?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Credit",
        "difficulty": "easy",
        "topic": "Credit requirements"
    },
    {
        "question": "How many credits are allocated to a 3-hour lecture course per week?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Credit",
        "difficulty": "easy",
        "topic": "Credit requirements"
    },
    {
        "question": "What is the credit value assigned to a 2-hour practical lab class per week?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Credit",
        "difficulty": "medium",
        "topic": "Credit requirements"
    },
    {
        "question": "What is the minimum credit load a student can register for in a regular semester?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Registration",
        "difficulty": "medium",
        "topic": "Credit requirements"
    },
    {
        "question": "What is the maximum credit load allowed to be registered by a student in a semester?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Registration",
        "difficulty": "medium",
        "topic": "Credit requirements"
    },
    {
        "question": "How many credits are allotted for the B.Tech Major Project?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Credit",
        "difficulty": "hard",
        "topic": "Credit requirements"
    },

    # 8. Minor
    {
        "question": "What is a Minor degree and who is eligible to register for it?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Minor",
        "difficulty": "easy",
        "topic": "Minor"
    },
    {
        "question": "How many additional credits must a student earn to qualify for a Minor degree?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Minor",
        "difficulty": "medium",
        "topic": "Minor"
    },
    {
        "question": "What is the minimum CGPA required to register for a Minor degree program?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Minor",
        "difficulty": "medium",
        "topic": "Minor"
    },
    {
        "question": "Can a student withdraw from a Minor degree program halfway through?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Minor",
        "difficulty": "medium",
        "topic": "Minor"
    },
    {
        "question": "Are the marks/grades of Minor courses included in the main CGPA calculation?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Minor",
        "difficulty": "hard",
        "topic": "Minor"
    },
    {
        "question": "What happens if a student completes only a few of the registered Minor courses?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Minor",
        "difficulty": "hard",
        "topic": "Minor"
    },

    # 9. Honors
    {
        "question": "What is an Honors degree and how does it differ from a regular B.Tech degree?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Honors",
        "difficulty": "easy",
        "topic": "Honors"
    },
    {
        "question": "How many extra credits are required to receive a B.Tech with Honors?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Honors",
        "difficulty": "easy",
        "topic": "Honors"
    },
    {
        "question": "What CGPA must a student maintain to register for the Honors program?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Honors",
        "difficulty": "medium",
        "topic": "Honors"
    },
    {
        "question": "Can a student register for both Honors and Minor degrees at Sreenidhi University?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Honors",
        "difficulty": "medium",
        "topic": "Honors"
    },
    {
        "question": "What happens to the Honors registration if a student's CGPA drops below the threshold?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Honors",
        "difficulty": "hard",
        "topic": "Honors"
    },
    {
        "question": "Are Honors courses offered through online portals like NPTEL or only in-person?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Honors",
        "difficulty": "hard",
        "topic": "Honors"
    },

    # 10. Course Registration
    {
        "question": "When does course registration typically take place for a semester?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Registration",
        "difficulty": "easy",
        "topic": "Course Registration"
    },
    {
        "question": "What is the procedure for registering for backlog courses?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Registration",
        "difficulty": "medium",
        "topic": "Course Registration"
    },
    {
        "question": "Who acts as a counselor/advisor for students during course registration?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Registration",
        "difficulty": "medium",
        "topic": "Course Registration"
    },
    {
        "question": "Can a student register for classes if they have unpaid college tuition fees?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Registration",
        "difficulty": "medium",
        "topic": "Course Registration"
    },
    {
        "question": "What happens if a student misses the course registration deadline?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Registration",
        "difficulty": "hard",
        "topic": "Course Registration"
    },
    {
        "question": "Are there credit prerequisites for registering for high-level core elective courses?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Registration",
        "difficulty": "hard",
        "topic": "Course Registration"
    },

    # 11. Add/Drop
    {
        "question": "What is the Add/Drop policy for registered courses?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Registration",
        "difficulty": "easy",
        "topic": "Add/Drop"
    },
    {
        "question": "How many weeks from the start of the semester do I have to add or drop a course?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Registration",
        "difficulty": "easy",
        "topic": "Add/Drop"
    },
    {
        "question": "Can I drop a course if my total registered credits fall below the minimum limit?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Registration",
        "difficulty": "medium",
        "topic": "Add/Drop"
    },
    {
        "question": "Does a course dropped during the Add/Drop period show up in the grade card?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Registration",
        "difficulty": "medium",
        "topic": "Add/Drop"
    },
    {
        "question": "What is the process to officially request adding a course?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Registration",
        "difficulty": "hard",
        "topic": "Add/Drop"
    },

    # 12. Withdrawal
    {
        "question": "What is the withdrawal policy for courses at Sreenidhi University?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Registration",
        "difficulty": "easy",
        "topic": "Withdrawal"
    },
    {
        "question": "What is the deadline or period within which a student can withdraw from a course?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Registration",
        "difficulty": "medium",
        "topic": "Withdrawal"
    },
    {
        "question": "How does withdrawing from a course differ from dropping a course?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Registration",
        "difficulty": "medium",
        "topic": "Withdrawal"
    },
    {
        "question": "Will a withdrawn course have a grade point, and will it be listed as 'W' grade?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Registration",
        "difficulty": "medium",
        "topic": "Withdrawal"
    },
    {
        "question": "Does withdrawing from a course affect my eligibility for Honors?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Registration",
        "difficulty": "hard",
        "topic": "Withdrawal"
    },

    # 13. Leave policy
    {
        "question": "What is the leave policy for students under academic regulations?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Leave",
        "difficulty": "easy",
        "topic": "Leave policy"
    },
    {
        "question": "How many days of leave can a student take in a semester without affecting attendance?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Leave",
        "difficulty": "medium",
        "topic": "Leave policy"
    },
    {
        "question": "Who is the authority that grants approval for general student leave?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Leave",
        "difficulty": "medium",
        "topic": "Leave policy"
    },
    {
        "question": "Are there leave benefits for participating in university-approved hackathons or seminars?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Leave",
        "difficulty": "hard",
        "topic": "Leave policy"
    },
    {
        "question": "What is the process to apply for academic leave at Sreenidhi University?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Leave",
        "difficulty": "hard",
        "topic": "Leave policy"
    },

    # 14. Medical leave
    {
        "question": "What is the medical leave policy for students at Sreenidhi?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Leave",
        "difficulty": "easy",
        "topic": "Medical leave"
    },
    {
        "question": "What supporting documents must be submitted to claim medical leave?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Leave",
        "difficulty": "easy",
        "topic": "Medical leave"
    },
    {
        "question": "Within how many days of returning from illness must I submit the medical certificate?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Leave",
        "difficulty": "medium",
        "topic": "Medical leave"
    },
    {
        "question": "Can medical leave be used to condone attendance below 65%?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Leave",
        "difficulty": "medium",
        "topic": "Medical leave"
    },
    {
        "question": "Who reviews and approves medical certificates submitted by students?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Leave",
        "difficulty": "hard",
        "topic": "Medical leave"
    },
    {
        "question": "Can a student write mid-term examinations separately if they were on medical leave?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Leave",
        "difficulty": "hard",
        "topic": "Medical leave"
    },

    # 15. Anti-ragging
    {
        "question": "What is Sreenidhi University's stance on ragging?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Discipline",
        "difficulty": "easy",
        "topic": "Anti-ragging"
    },
    {
        "question": "What are the disciplinary actions/punishments for students caught ragging?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Discipline",
        "difficulty": "medium",
        "topic": "Anti-ragging"
    },
    {
        "question": "Who should a student contact to report ragging incident on campus?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Discipline",
        "difficulty": "medium",
        "topic": "Anti-ragging"
    },
    {
        "question": "Is an anti-ragging affidavit required to be signed by students and parents during admission?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Discipline",
        "difficulty": "medium",
        "topic": "Anti-ragging"
    },
    {
        "question": "What is the role of the Anti-Ragging Committee at the university?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Discipline",
        "difficulty": "hard",
        "topic": "Anti-ragging"
    },
    {
        "question": "Can ragging result in permanent expulsion from the university and legal action?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Discipline",
        "difficulty": "hard",
        "topic": "Anti-ragging"
    },

    # 16. Academic calendar
    {
        "question": "When does the ODD semester for A24 V and A25 III start according to the Academic Calendar?",
        "expected_document": "Academic Calendar 2026-27 A24 V and A25 III ODD Semesters.pdf",
        "expected_section": "Academic Calendar",
        "difficulty": "easy",
        "topic": "Academic calendar"
    },
    {
        "question": "When are the Mid-I examinations scheduled for the ODD semester in academic calendar?",
        "expected_document": "Academic Calendar 2026-27 A24 V and A25 III ODD Semesters.pdf",
        "expected_section": "Academic Calendar",
        "difficulty": "medium",
        "topic": "Academic calendar"
    },
    {
        "question": "What are the start and end dates of the Semester End Examinations for EVEN semesters?",
        "expected_document": "Academic Calendar 2026-27 A24 VI and A25 IV EVEN Semesters.pdf",
        "expected_section": "Academic Calendar",
        "difficulty": "medium",
        "topic": "Academic calendar"
    },
    {
        "question": "How many instructional weeks are scheduled for the EVEN semester?",
        "expected_document": "Academic Calendar 2026-27 A24 VI and A25 IV EVEN Semesters.pdf",
        "expected_section": "Academic Calendar",
        "difficulty": "medium",
        "topic": "Academic calendar"
    },
    {
        "question": "When are the summer vacation/recess dates scheduled in the calendar?",
        "expected_document": "Academic Calendar 2026-27 A24 VI and A25 IV EVEN Semesters.pdf",
        "expected_section": "Academic Calendar",
        "difficulty": "hard",
        "topic": "Academic calendar"
    },
    {
        "question": "How many total instruction days are guaranteed in a semester as per the academic calendar?",
        "expected_document": "Academic Calendar 2026-27 A24 V and A25 III ODD Semesters.pdf",
        "expected_section": "Academic Calendar",
        "difficulty": "hard",
        "topic": "Academic calendar"
    },

    # 17. Internship
    {
        "question": "What is the policy for mandatory student internships under R26?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Internship",
        "difficulty": "easy",
        "topic": "Internship"
    },
    {
        "question": "When are students eligible or expected to undertake industrial internships?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Internship",
        "difficulty": "easy",
        "topic": "Internship"
    },
    {
        "question": "How many credits are allocated for internship evaluation?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Internship",
        "difficulty": "medium",
        "topic": "Internship"
    },
    {
        "question": "Is a full-semester internship allowed in the final year of the B.Tech program?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Internship",
        "difficulty": "medium",
        "topic": "Internship"
    },
    {
        "question": "What documents/reports must be submitted to the department after completing internship?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Internship",
        "difficulty": "hard",
        "topic": "Internship"
    },
    {
        "question": "Who evaluates the internship, and is there an external viva voce?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Internship",
        "difficulty": "hard",
        "topic": "Internship"
    },

    # 18. Promotion
    {
        "question": "What are the promotion rules from the 1st year to the 2nd year B.Tech?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Promotion",
        "difficulty": "easy",
        "topic": "Promotion"
    },
    {
        "question": "How many credits must a student earn to be promoted from the 2nd year to the 3rd year?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Promotion",
        "difficulty": "medium",
        "topic": "Promotion"
    },
    {
        "question": "What is the credit requirement for promotion to the 4th year?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Promotion",
        "difficulty": "medium",
        "topic": "Promotion"
    },
    {
        "question": "What happens if a student fails to acquire the necessary credits for academic promotion?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Promotion",
        "difficulty": "hard",
        "topic": "Promotion"
    },
    {
        "question": "Are promotion requirements different for lateral entry students?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Promotion",
        "difficulty": "hard",
        "topic": "Promotion"
    },

    # 19. Revaluation
    {
        "question": "What is the procedure to apply for revaluation of exam answer scripts?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Revaluation",
        "difficulty": "easy",
        "topic": "Revaluation"
    },
    {
        "question": "What is the timeframe within which a student must apply for revaluation after results are out?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Revaluation",
        "difficulty": "easy",
        "topic": "Revaluation"
    },
    {
        "question": "Is there a fee for applying for course revaluation?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Revaluation",
        "difficulty": "medium",
        "topic": "Revaluation"
    },
    {
        "question": "Does the revaluation grade replace the old grade even if the marks decrease?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Revaluation",
        "difficulty": "medium",
        "topic": "Revaluation"
    },
    {
        "question": "What is the difference between recounting and revaluation?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Revaluation",
        "difficulty": "hard",
        "topic": "Revaluation"
    },
    {
        "question": "Under what condition is a third valuation done during revaluation?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Revaluation",
        "difficulty": "hard",
        "topic": "Revaluation"
    },

    # 20. Branch change
    {
        "question": "Is there an option for a change of branch after the first year?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Branch Change",
        "difficulty": "easy",
        "topic": "Branch change"
    },
    {
        "question": "When can a student apply for a branch change in B.Tech?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Branch Change",
        "difficulty": "easy",
        "topic": "Branch change"
    },
    {
        "question": "What is the eligibility criteria (e.g., minimum CGPA) for change of branch?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Branch Change",
        "difficulty": "medium",
        "topic": "Branch change"
    },
    {
        "question": "How is branch change allocated if multiple students apply for the same vacant seat?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Branch Change",
        "difficulty": "medium",
        "topic": "Branch change"
    },
    {
        "question": "Can a student admitted under lateral entry request a branch change?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Branch Change",
        "difficulty": "hard",
        "topic": "Branch change"
    },
    {
        "question": "What happens to the credits and courses of a student who successfully changes their branch?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Branch Change",
        "difficulty": "hard",
        "topic": "Branch change"
    },

    # 21. Semester duration
    {
        "question": "What is the standard duration of a semester in terms of weeks?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Academic Calendar",
        "difficulty": "easy",
        "topic": "Semester duration"
    },
    {
        "question": "How many total instruction days must a semester contain as a minimum?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Academic Calendar",
        "difficulty": "medium",
        "topic": "Semester duration"
    },
    {
        "question": "How many hours of teaching/instruction correspond to a 1-credit theory course?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Credit",
        "difficulty": "medium",
        "topic": "Semester duration"
    },
    {
        "question": "What is the duration of preparation holidays before Semester End Examinations?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Academic Calendar",
        "difficulty": "hard",
        "topic": "Semester duration"
    },

    # 22. Curriculum
    {
        "question": "What courses are prescribed in the B.Tech CSE curriculum for the 3rd year?",
        "expected_document": "A24_BTech_CSE_Curriculum_Syllabus_03July2026.pdf",
        "expected_section": "Curriculum",
        "difficulty": "medium",
        "topic": "Curriculum"
    },
    {
        "question": "What are the core subjects taught in B.Tech AIML in the 5th semester?",
        "expected_document": "A24_BTech_AIML_Curriculum_Syllabus_03July2026.pdf",
        "expected_section": "Curriculum",
        "difficulty": "medium",
        "topic": "Curriculum"
    },
    {
        "question": "What is the credit weighting of professional electives vs open electives in CSE syllabus?",
        "expected_document": "A24_BTech_CSE_Curriculum_Syllabus_03July2026.pdf",
        "expected_section": "Curriculum",
        "difficulty": "hard",
        "topic": "Curriculum"
    },
    {
        "question": "How are professional core courses structured in the AIML curriculum?",
        "expected_document": "A24_BTech_AIML_Curriculum_Syllabus_03July2026.pdf",
        "expected_section": "Curriculum",
        "difficulty": "hard",
        "topic": "Curriculum"
    },
    {
        "question": "Does the CSE curriculum include non-credit mandatory courses?",
        "expected_document": "A24_BTech_CSE_Curriculum_Syllabus_03July2026.pdf",
        "expected_section": "Curriculum",
        "difficulty": "hard",
        "topic": "Curriculum"
    },

    # 23. Credit transfer
    {
        "question": "Can credits earned from online courses like NPTEL or MOOCs be transferred?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Credit Transfer",
        "difficulty": "easy",
        "topic": "Credit transfer"
    },
    {
        "question": "What is the maximum limit of credits that can be transferred via online courses?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Credit Transfer",
        "difficulty": "medium",
        "topic": "Credit transfer"
    },
    {
        "question": "Who must approve the choice of NPTEL/MOOCs course for credit transfer?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Credit Transfer",
        "difficulty": "medium",
        "topic": "Credit transfer"
    },
    {
        "question": "How are the letter grades mapped for transferred credits from MOOCs?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Credit Transfer",
        "difficulty": "hard",
        "topic": "Credit transfer"
    },
    {
        "question": "What happens if a student registers for credit transfer but fails the online exam?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Credit Transfer",
        "difficulty": "hard",
        "topic": "Credit transfer"
    },

    # 24. FAQs
    {
        "question": "What is the passing marks percentage in a theory course?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Evaluation",
        "difficulty": "easy",
        "topic": "FAQs"
    },
    {
        "question": "Can I write exams if I lose my physical Hall Ticket?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Examinations",
        "difficulty": "easy",
        "topic": "FAQs"
    },
    {
        "question": "What is the penalty for copying/malpractice in a midterm examination?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Discipline",
        "difficulty": "medium",
        "topic": "FAQs"
    },
    {
        "question": "How do I get my official academic transcript from Sreenidhi University?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "General",
        "difficulty": "medium",
        "topic": "FAQs"
    },
    {
        "question": "Who is the chief authority of the examination branch at Sreenidhi?",
        "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf",
        "expected_section": "Examinations",
        "difficulty": "hard",
        "topic": "FAQs"
    }
]

# We need to expand this catalog to at least 150 questions. Let's dynamically add variants to cover all topics fully.
# Let's add variants for each of the 24 topics to reach at least 150 questions (e.g. 6-7 questions per topic).
# Currently we have 6 + 7 + 6 + 6 + 5 + 6 + 6 + 6 + 6 + 6 + 5 + 5 + 5 + 6 + 6 + 6 + 6 + 5 + 6 + 6 + 4 + 5 + 5 + 5 = 134 questions.
# Let's add 28 additional questions to reach 162 questions.
additional_questions = [
    # Attendance additions
    {"question": "How is attendance computed for lab-integrated theory courses?", "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf", "expected_section": "Attendance", "difficulty": "hard", "topic": "Attendance"},
    {"question": "Can attendance be condoned for pregnancy/long-term medical illness?", "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf", "expected_section": "Attendance", "difficulty": "hard", "topic": "Attendance"},
    
    # CIE additions
    {"question": "Is there any assignment component in laboratory CIE marks?", "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf", "expected_section": "Evaluation", "difficulty": "medium", "topic": "CIE"},
    {"question": "Who moderates CIE marks in case of discrepancies?", "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf", "expected_section": "Evaluation", "difficulty": "hard", "topic": "CIE"},

    # SEE additions
    {"question": "What is the passing criteria for a student in a laboratory-integrated course?", "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf", "expected_section": "Evaluation", "difficulty": "hard", "topic": "SEE"},

    # Grades additions
    {"question": "What is the grade point for an 'A+' grade in regulations?", "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf", "expected_section": "Grading", "difficulty": "easy", "topic": "Grades"},

    # SGPA/CGPA additions
    {"question": "How does failing a non-credit course affect my SGPA or CGPA?", "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf", "expected_section": "Grading", "difficulty": "medium", "topic": "SGPA"},
    {"question": "Are grades from summer courses included in CGPA?", "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf", "expected_section": "Grading", "difficulty": "medium", "topic": "CGPA"},

    # Credit additions
    {"question": "How are credits distributed between semesters in 1st year B.Tech?", "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf", "expected_section": "Credit", "difficulty": "easy", "topic": "Credit requirements"},
    
    # Minor/Honors additions
    {"question": "Can I register for a Minor in my own parent department?", "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf", "expected_section": "Minor", "difficulty": "medium", "topic": "Minor"},
    {"question": "Is there a limit on class size for Honors course registration?", "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf", "expected_section": "Honors", "difficulty": "hard", "topic": "Honors"},

    # Registration additions
    {"question": "What is the maximum backlog credits a student can register in a single semester?", "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf", "expected_section": "Registration", "difficulty": "hard", "topic": "Course Registration"},

    # Add/Drop/Withdrawal additions
    {"question": "Can a freshman student drop a math course in the first semester?", "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf", "expected_section": "Registration", "difficulty": "medium", "topic": "Add/Drop"},
    {"question": "Is there any academic fee refund if I withdraw from a course?", "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf", "expected_section": "Registration", "difficulty": "hard", "topic": "Withdrawal"},

    # Leave additions
    {"question": "Are duty leaves counted under academic leave for attendance calculation?", "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf", "expected_section": "Leave", "difficulty": "medium", "topic": "Leave policy"},
    {"question": "What happens if a student takes unapproved leave of more than 15 consecutive days?", "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf", "expected_section": "Leave", "difficulty": "hard", "topic": "Leave policy"},

    # Medical additions
    {"question": "Does the medical leave claim require endorsement from the college doctor?", "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf", "expected_section": "Leave", "difficulty": "medium", "topic": "Medical leave"},

    # Anti-ragging additions
    {"question": "Can a student be suspended immediately upon a ragging complaint?", "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf", "expected_section": "Discipline", "difficulty": "medium", "topic": "Anti-ragging"},

    # Calendar additions
    {"question": "Where is the academic calendar published for students to access?", "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf", "expected_section": "Academic Calendar", "difficulty": "easy", "topic": "Academic calendar"},
    {"question": "Does the academic calendar outline parent-teacher meeting dates?", "expected_document": "Academic Calendar 2026-27 A24 V and A25 III ODD Semesters.pdf", "expected_section": "Academic Calendar", "difficulty": "easy", "topic": "Academic calendar"},

    # Internship additions
    {"question": "Can I do an internship during the winter break between semesters?", "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf", "expected_section": "Internship", "difficulty": "easy", "topic": "Internship"},

    # Promotion additions
    {"question": "How are credits defined as acquired vs registered for promotion?", "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf", "expected_section": "Promotion", "difficulty": "hard", "topic": "Promotion"},

    # Revaluation additions
    {"question": "Can I get a copy of my evaluated answer script along with revaluation?", "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf", "expected_section": "Revaluation", "difficulty": "medium", "topic": "Revaluation"},

    # Branch change additions
    {"question": "Does branch change affect the hostel or fee structure of the student?", "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf", "expected_section": "Branch Change", "difficulty": "easy", "topic": "Branch change"},

    # Semester duration additions
    {"question": "Are exam weeks counted in the instruction weeks of a semester?", "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf", "expected_section": "Academic Calendar", "difficulty": "medium", "topic": "Semester duration"},

    # Curriculum additions
    {"question": "What is the syllabus of Data Structures in CSE or AIML first year?", "expected_document": "A24_BTech_CSE_Curriculum_Syllabus_03July2026.pdf", "expected_section": "Curriculum", "difficulty": "hard", "topic": "Curriculum"},
    {"question": "What are the course outcomes listed for AIML foundation course?", "expected_document": "A24_BTech_AIML_Curriculum_Syllabus_03July2026.pdf", "expected_section": "Curriculum", "difficulty": "hard", "topic": "Curriculum"},

    # Credit transfer additions
    {"question": "Is credit transfer allowed from foreign universities under R26?", "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf", "expected_section": "Credit Transfer", "difficulty": "hard", "topic": "Credit transfer"},

    # FAQs additions
    {"question": "What is the procedure to clear an 'F' grade in subsequent semesters?", "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf", "expected_section": "Evaluation", "difficulty": "medium", "topic": "FAQs"},
    {"question": "What is the role of the Dean of Academics at Sreenidhi?", "expected_document": "R26_Rules_Regulations_v1.0_03July2026.pdf", "expected_section": "General", "difficulty": "medium", "topic": "FAQs"}
]

QUESTIONS_CATALOG.extend(additional_questions)

def main():
    logger.info(f"Loaded {len(QUESTIONS_CATALOG)} question templates.")
    
    # Initialize services
    try:
        embedder_service = EmbedderService()
        retriever_service = RetrieverService(embedder_service)
        llm_service = LLMService()
    except Exception as e:
        logger.error(f"Failed to initialize core backend services: {e}")
        logger.error("Please make sure backend/.env has valid PINECONE_API_KEY and GROQ_API_KEY.")
        sys.exit(1)
        
    dataset = []
    
    for idx, item in enumerate(QUESTIONS_CATALOG):
        q_id = idx + 1
        q_text = item["question"]
        expected_doc = item["expected_document"]
        expected_sec = item["expected_section"]
        diff = item["difficulty"]
        topic = item["topic"]
        
        logger.info(f"[{q_id}/{len(QUESTIONS_CATALOG)}] Generating ground truth for question: '{q_text}'")
        
        try:
            # 1. Retrieve context chunks using local retriever
            retrieved_chunks = retriever_service.retrieve(q_text, top_k=3)
            
            # Combine the texts
            context_texts = [c.get("text", "") for c in retrieved_chunks]
            context_combined = "\n\n".join(context_texts)
            
            # 2. Query Groq to generate a highly grounded ground_truth answer
            system_prompt = (
                "You are Sreenidhi University's Academic Regulations Specialist.\n"
                "Your task is to write a highly accurate, clear, and direct answer to the user's question, "
                "based ONLY on the provided context. If the context does not contain the answer, "
                "say 'Information is not available in the regulations.'\n"
                "Keep the answer concise (2-4 sentences max).\n\n"
                f"Context:\n{context_combined}"
            )
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q_text}
            ]
            
            ground_truth = llm_service.generate_answer(messages)
            ground_truth = ground_truth.strip()
            
            # Save test case
            dataset.append({
                "id": q_id,
                "question": q_text,
                "ground_truth": ground_truth,
                "expected_document": expected_doc,
                "expected_section": expected_sec,
                "difficulty": diff,
                "topic": topic
            })
            
            # Rate limit mitigation for Groq if needed
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Error generating ground truth for question ID {q_id}: {e}")
            # Fallback ground truth
            dataset.append({
                "id": q_id,
                "question": q_text,
                "ground_truth": "Failed to generate ground truth answer due to API error.",
                "expected_document": expected_doc,
                "expected_section": expected_sec,
                "difficulty": diff,
                "topic": topic
            })

    # Save to dataset.json
    output_path = os.path.join(os.path.dirname(__file__), "dataset.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)
        
    logger.info(f"Successfully generated {len(dataset)} evaluation questions in '{output_path}'.")

if __name__ == "__main__":
    main()
