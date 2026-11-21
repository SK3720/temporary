#pragma once
// Snap a spelled word to the closest common first name within one edit.
// Substitutions between letters the CNN is known to confuse cost less, so
// "JOHM" -> "JOHN" (M/N) beats an arbitrary one-letter change.
#include <string.h>
#include "names.h"

static const char* const kConfusable[] = {"AEMNST", "UVR", "GH", "IJ", "DZ", "KV"};

static bool confusable(char a, char b) {
  for (const char* g : kConfusable)
    if (strchr(g, a) && strchr(g, b)) return true;
  return false;
}

// Cost of turning `w` into `n` if it is within one edit, else -1.
// 0 = identical, 1 = confusable substitution, 2 = any other single edit.
static int edit1_cost(const char* w, const char* n) {
  int lw = strlen(w), ln = strlen(n);
  if (lw == ln) {
    int diff = -1;
    for (int i = 0; i < lw; i++)
      if (w[i] != n[i]) { if (diff >= 0) return -1; diff = i; }
    if (diff < 0) return 0;
    return confusable(w[diff], n[diff]) ? 1 : 2;
  }
  if (abs(lw - ln) != 1) return -1;
  const char* s = lw < ln ? w : n;   // shorter
  const char* l = lw < ln ? n : w;   // longer
  int i = 0, j = 0;
  bool skipped = false;
  while (s[i] && l[j]) {
    if (s[i] == l[j]) { i++; j++; continue; }
    if (skipped) return -1;
    skipped = true;
    j++;
  }
  return 2;
}

// Returns the index into kNames of the suggested name, or -1 if the word is
// already a known name, too short, or nothing is within one edit.
static int snap_to_name(const char* word) {
  if ((int)strlen(word) < SNAP_MIN_LEN) return -1;
  int best = -1, bestCost = 99;
  for (int i = 0; i < kNumNames; i++) {
    int c = edit1_cost(word, kNames[i]);
    if (c == 0) return -1;                  // exact match: no correction
    if (c > 0 && c < bestCost) { best = i; bestCost = c; }   // list is most-common-first
  }
  return best;
}
