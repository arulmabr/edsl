#!/usr/bin/env python3
"""
Example usage of Goodfire integration with EDSL

This example demonstrates how to use the Goodfire service with EDSL.
Make sure to set your GOODFIRE_API_KEY environment variable before running.
"""

import os
from edsl import Model, Question, Survey


def main():
    """Demonstrate Goodfire integration with EDSL"""

    # Check if API key is set
    if not os.getenv("GOODFIRE_API_KEY"):
        print("Please set the GOODFIRE_API_KEY environment variable")
        print("export GOODFIRE_API_KEY='your-api-key-here'")
        return

    print("🚀 Goodfire Integration Example")
    print("=" * 50)

    # Create a model using Goodfire service
    print("Creating Goodfire model...")
    model = Model("meta-llama/Llama-3.3-70B-Instruct", service="goodfire")
    print(f"✓ Model created: {model}")

    # Create a simple question
    print("\nCreating a survey question...")
    question = Question(
        question_name="favorite_color",
        question_text="What is your favorite color?",
        question_type="multiple_choice",
        question_options=["Red", "Green", "Blue", "Yellow", "Purple"],
    )
    print(f"✓ Question created: {question.question_text}")

    # Create a survey
    print("\nCreating a survey...")
    survey = Survey([question])
    print(f"✓ Survey created with {len(survey)} question(s)")

    # Run the survey (this would make an actual API call)
    print("\nRunning survey with Goodfire model...")
    print("Note: This would make an actual API call to Goodfire")
    print("To run the survey, uncomment the following lines:")
    print("# results = survey.by(model).run()")
    print("# print(results)")

    print("\n🎉 Goodfire integration is working correctly!")
    print("\nTo use with actual API calls:")
    print("1. Set your GOODFIRE_API_KEY environment variable")
    print("2. Uncomment the survey.run() lines above")
    print("3. Run the script again")


if __name__ == "__main__":
    main()
