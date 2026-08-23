"""
End-to-end test for MemoOS.

    python test_memoos.py

Creates two users, fills both with memories, and checks that search
ranks sensibly, that edits re-embed, that deletes stick — and above all
that neither user can see or touch the other's memories, including by
guessing a real memory id.

Runs against a scratch database so it never touches memoos.db.
"""

import os
import sys
import tempfile

os.environ.setdefault("MEMOOS_DB_PATH", os.path.join(tempfile.mkdtemp(), "test.db"))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

ALICE, BOB = "alice", "bob"

ALICE_MEMORIES = [
    "Alice lives in Bengaluru and works as a backend engineer at Swiggy.",
    "Alice's favourite programming language is Go, though she learned Python first.",
    "Alice has a golden retriever named Rex.",
    "Alice is allergic to peanuts.",
    "Alice is training for a half marathon in December.",
]

BOB_MEMORIES = [
    "Bob lives in Toronto and teaches high school chemistry.",
    "Bob plays bass guitar in a jazz trio on weekends.",
    "Bob is saving up to buy a used Subaru.",
]

passed = failed = 0


def check(label, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {label}")
    else:
        failed += 1
        print(f"  FAIL  {label}" + (f"\n          {detail}" if detail else ""))


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def main() -> int:
    with TestClient(app) as client:

        section("health")
        r = client.get("/health")
        body = r.json()
        print(f"  ollama={body['ollama']} database={body['database']} "
              f"model={body['embed_model']}")
        check("health responds", r.status_code == 200, r.text[:200])
        if body["ollama"] != "ok":
            print(f"\n  Ollama is not usable: {body.get('detail')}")
            print("  Start it with `ollama serve` and pull the model:")
            print("      ollama pull nomic-embed-text")
            return 1

        section("add memories for two users")
        alice_ids, bob_ids = [], []
        for text in ALICE_MEMORIES:
            r = client.post(f"/users/{ALICE}/memories", json={"content": text})
            if r.status_code != 201:
                check("add succeeded", False, r.text[:300])
                return 1
            alice_ids.append(r.json()["created"][0]["id"])
        for text in BOB_MEMORIES:
            r = client.post(f"/users/{BOB}/memories", json={"content": text})
            bob_ids.append(r.json()["created"][0]["id"])
        check(f"stored {len(alice_ids)} for alice", len(alice_ids) == 5)
        check(f"stored {len(bob_ids)} for bob", len(bob_ids) == 3)

        section("implicit provisioning")
        r = client.post("/users/brand-new-user/memories",
                        json={"content": "This user did not exist a second ago."})
        check("unknown user_id just works", r.status_code == 201, r.text[:200])
        r = client.get("/users/brand-new-user/memories")
        check("and has their own space", r.json()["total"] == 1, r.text[:200])

        section("list + pagination")
        r = client.get(f"/users/{ALICE}/memories")
        check("alice sees 5", r.json()["total"] == 5, r.text[:200])
        r = client.get(f"/users/{BOB}/memories")
        check("bob sees 3", r.json()["total"] == 3, r.text[:200])
        r = client.get(f"/users/{ALICE}/memories", params={"limit": 2, "offset": 0})
        page1 = r.json()
        check("page size respected", len(page1["memories"]) == 2, r.text[:200])
        check("total is the full count, not the page", page1["total"] == 5, page1["total"])
        r = client.get(f"/users/{ALICE}/memories", params={"limit": 2, "offset": 2})
        page2 = r.json()
        ids1 = {m["id"] for m in page1["memories"]}
        ids2 = {m["id"] for m in page2["memories"]}
        check("pages do not overlap", ids1.isdisjoint(ids2), f"{ids1 & ids2}")

        section("semantic search ranks sensibly")
        r = client.get(f"/users/{ALICE}/memories/search",
                       params={"q": "does she have any pets?", "top_k": 3})
        hits = r.json()["hits"]
        check("search returns hits", len(hits) > 0, r.text[:300])
        top = hits[0]["memory"]["content"]
        print(f"    'does she have any pets?' -> {top!r}  ({hits[0]['score']:.3f})")
        check("pet question finds the dog", "Rex" in top, top)
        check("scores are ordered", all(
            hits[i]["score"] >= hits[i + 1]["score"] for i in range(len(hits) - 1)),
            [h["score"] for h in hits])
        check("cosine stays in range", all(-1.0 <= h["score"] <= 1.0 for h in hits),
              [h["score"] for h in hits])

        r = client.get(f"/users/{ALICE}/memories/search",
                       params={"q": "what food should she avoid?", "top_k": 2})
        top = r.json()["hits"][0]["memory"]["content"]
        print(f"    'what food should she avoid?' -> {top!r}")
        check("allergy question finds the allergy", "peanut" in top.lower(), top)

        r = client.get(f"/users/{ALICE}/memories/search",
                       params={"q": "where does she work?", "top_k": 2})
        top = r.json()["hits"][0]["memory"]["content"]
        print(f"    'where does she work?' -> {top!r}")
        check("job question finds the employer", "Swiggy" in top, top)

        section("top_k is honoured")
        r = client.get(f"/users/{ALICE}/memories/search", params={"q": "alice", "top_k": 2})
        check("top_k caps results", len(r.json()["hits"]) == 2, len(r.json()["hits"]))

        section("ISOLATION — the critical rule")
        r = client.get(f"/users/{BOB}/memories/search",
                       params={"q": "does she have any pets?", "top_k": 5})
        bob_hits = [h["memory"]["content"] for h in r.json()["hits"]]
        check("bob's search never returns alice's memories",
              all("Alice" not in c for c in bob_hits), bob_hits)
        check("bob's search is capped at bob's own count", len(bob_hits) <= 3, bob_hits)

        r = client.get(f"/users/{ALICE}/memories/search",
                       params={"q": "jazz bass guitar Subaru Toronto", "top_k": 5})
        alice_hits = [h["memory"]["content"] for h in r.json()["hits"]]
        check("alice's search never returns bob's memories",
              all("Bob" not in c for c in alice_hits), alice_hits)

        stolen = alice_ids[0]
        r = client.get(f"/users/{BOB}/memories/{stolen}")
        check("bob cannot GET alice's memory by id", r.status_code == 404, r.status_code)
        r = client.patch(f"/users/{BOB}/memories/{stolen}", json={"content": "hijacked"})
        check("bob cannot PATCH alice's memory by id", r.status_code == 404, r.status_code)
        r = client.delete(f"/users/{BOB}/memories/{stolen}")
        check("bob cannot DELETE alice's memory by id", r.status_code == 404, r.status_code)

        r = client.get(f"/users/{ALICE}/memories/{stolen}")
        check("and alice's memory is untouched after all that",
              r.status_code == 200 and r.json()["content"] == ALICE_MEMORIES[0],
              r.text[:200])
        r = client.get(f"/users/{ALICE}/memories")
        check("alice still has all 5", r.json()["total"] == 5, r.json()["total"])

        section("edit re-embeds")
        target = alice_ids[2]  # the dog
        r = client.patch(f"/users/{ALICE}/memories/{target}",
                         json={"content": "Alice has a tabby cat named Mango."})
        check("patch succeeds", r.status_code == 200, r.text[:200])
        check("content changed", r.json()["content"].endswith("Mango."), r.json()["content"])
        check("updated_at moved", r.json()["updated_at"] >= r.json()["created_at"])

        r = client.get(f"/users/{ALICE}/memories/search",
                       params={"q": "does she have any pets?", "top_k": 1})
        top = r.json()["hits"][0]["memory"]["content"]
        print(f"    after edit, 'pets?' -> {top!r}")
        check("search reflects the edit, not the old text", "Mango" in top, top)

        r = client.patch(f"/users/{ALICE}/memories/{target}",
                         json={"metadata": {"confirmed": True}})
        check("metadata-only patch keeps content",
              r.json()["content"].endswith("Mango.") and
              r.json()["metadata"] == {"confirmed": True}, r.text[:200])

        section("delete")
        victim = alice_ids[4]
        r = client.delete(f"/users/{ALICE}/memories/{victim}")
        check("delete succeeds", r.status_code == 200, r.text[:200])
        r = client.get(f"/users/{ALICE}/memories/{victim}")
        check("deleted memory is gone", r.status_code == 404, r.status_code)
        r = client.get(f"/users/{ALICE}/memories")
        check("count dropped to 4", r.json()["total"] == 4, r.json()["total"])
        r = client.delete(f"/users/{ALICE}/memories/{victim}")
        check("deleting twice 404s", r.status_code == 404, r.status_code)
        r = client.get(f"/users/{ALICE}/memories/search",
                       params={"q": "half marathon training", "top_k": 5})
        check("deleted memory is out of the index",
              all("marathon" not in h["memory"]["content"] for h in r.json()["hits"]),
              [h["memory"]["content"] for h in r.json()["hits"]])

        section("edges")
        r = client.post(f"/users/{ALICE}/memories", json={"content": "   "})
        check("blank content rejected", r.status_code == 422, r.status_code)
        r = client.get("/users/nobody-at-all/memories")
        check("unknown user lists empty", r.json()["total"] == 0, r.text[:200])
        r = client.get("/users/nobody-at-all/memories/search", params={"q": "anything"})
        check("unknown user search is empty, not an error",
              r.status_code == 200 and r.json()["hits"] == [], r.text[:200])

        section("fact extraction (optional path)")
        raw = ("hey so I finally moved to Pune last week, joined Zomato as a "
               "data engineer, and honestly I still prefer Python over Go")
        r = client.post(f"/users/{ALICE}/memories",
                        json={"content": raw, "extract": True})
        body = r.json()
        check("extraction request succeeds", r.status_code == 201, r.text[:300])
        if body.get("extracted"):
            print(f"    raw: {raw!r}")
            for m in body["created"]:
                print(f"    ->  {m['content']!r}")
            check("produced at least one fact", len(body["created"]) >= 1)
            check("facts are shorter than the ramble",
                  all(len(m["content"]) < len(raw) for m in body["created"]),
                  [m["content"] for m in body["created"]])
            check("provenance kept",
                  all(m["metadata"].get("source_text") == raw for m in body["created"]),
                  body["created"][0]["metadata"])
        else:
            print("    model returned nothing usable; fell back to raw text")
            check("fallback stored the raw text",
                  body["created"][0]["content"] == raw, body["created"][0]["content"])

    print(f"\n{'=' * 52}")
    print(f"  {passed} passed, {failed} failed")
    print(f"{'=' * 52}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
