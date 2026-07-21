# v0.8 output-blind author review

Complete pass 1 and pass 2 separately, at different times, without opening `hidden/reference-labels.json`, model outputs, or analysis files. Fill every review field. Use JSON-array syntax in plural fields. After both passes are complete, compare them, record every disagreement and its resolution, then update `author-review-status.json` with reviewer identity, timestamps, reconciliation evidence, and status `completed_two_author_passes`. The protocol freezer validates that status; it does not claim independent adjudication.
