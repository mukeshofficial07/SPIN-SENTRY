"""Debug script: run all three demo PDFs through the actual pipeline and print detailed results."""
import sys, os, json
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import run_pipeline

DEMO_PDFS = {
    "Safe":      "data/demo_papers/SPIN_SENTRY_Demo_Safe.pdf",
    "Risk":      "data/demo_papers/SPIN_SENTRY_Demo_Risk.pdf",
    "High_Risk": "data/demo_papers/SPIN_SENTRY_Demo_High_Risk.pdf",
}

RESULTS = {}

for label, path in DEMO_PDFS.items():
    print(f"\n{'='*60}")
    print(f"PDF: {label}  ({path})")
    print('='*60)
    if not os.path.exists(path):
        print(f"  !! FILE NOT FOUND: {path}")
        continue
    try:
        result = run_pipeline(path, top_k=3)
        s = result.summary
        print(f"  Total statements  : {s['total_claims']}")
        print(f"  Empirical claims  : {s['empirical_claims']}")
        print(f"  Protocol/Methods  : {s['protocol_statements']}")
        print(f"  Supported         : {s['supported_claims']}")
        print(f"  Flagged           : {s['flagged_claims']}")
        print(f"  Avg WE            : {s['avg_we_score']:.3f}")
        print(f"  Spin Risk         : {s['spin_risk']}")
        print(f"  Flag rate (emp)   : {s.get('flag_rate', 'N/A')}")
        print()
        for i, r in enumerate(result.reports):
            print(f"  -- Claim {i+1} --")
            print(f"     role       : {r.role}")
            print(f"     empirical  : {r.is_empirical_claim}")
            print(f"     types      : {r.claim_types}")
            print(f"     zone_route : {r.zone_route}")
            print(f"     WE score   : {r.we_score:.4f}")
            print(f"     forward NLI: {r.forward:.4f}")
            print(f"     reverse NLI: {r.reverse:.4f}")
            print(f"     contra NLI : {getattr(r, 'contradiction', 'N/A')}")
            print(f"     spin_flag  : {r.spin_flag}")
            print(f"     support    : {r.support_label}")
            print(f"     hedging    : {r.hedging}")
            print(f"     text       : {r.claim[:100]}")
            if r.evidence_items:
                ev = r.evidence_items[0]
                print(f"     top evidence: {str(ev)[:120]}")
            print()
        RESULTS[label] = result.to_dict()
    except Exception as e:
        import traceback
        print(f"  !! ERROR: {e}")
        traceback.print_exc()

# Save JSON results
out_path = "data/debug_pipeline_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print(f"\nSaved results to {out_path}")
