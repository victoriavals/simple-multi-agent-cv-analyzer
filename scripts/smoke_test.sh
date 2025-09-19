#!/bin/bash
# Smoke test for AI CV Analyzer
# Basic validation that the core pipeline works

set -e

echo "🔍 Running AI CV Analyzer smoke test..."
echo ""

# Test CLI with sample data
echo "✅ Testing CLI pipeline..."
python main.py --cv samples/sample_cv.txt --role "Senior AI Engineer" --language indonesia --provider auto --out samples/sample_output.md

echo ""
echo "✅ CLI test completed successfully!"
echo ""

# Streamlit hint (don't block)
echo "💡 To test Streamlit UI manually, run:"
echo "   streamlit run app.py"
echo ""
echo "🎉 Smoke test passed! Core functionality is working."