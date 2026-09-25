import unittest

from identity_graph import direct_response_events, self_identification_events


def row(index, cluster, text):
    return {"turn_idx": index, "cluster": cluster, "text": text}


class IdentityEventTest(unittest.TestCase):
    def test_self_identification(self):
        events = self_identification_events(
            "S1", [row(0, "SPK_001", "Hi, I'm Mia Chen.")]
        )
        self.assertEqual([(event["cluster"], event["name"]) for event in events], [("SPK_001", "Mia Chen")])

    def test_compound_leading_vocative(self):
        events = direct_response_events(
            "S1",
            [
                row(0, "SPK_001", "Oh, hey Marco, Leo here taking over your file."),
                row(1, "SPK_002", "No problem, Leo."),
            ],
        )
        self.assertEqual([(event["cluster"], event["name"]) for event in events], [("SPK_002", "Marco")])

    def test_trailing_vocative(self):
        events = direct_response_events(
            "S1",
            [
                row(0, "SPK_001", "Please continue, Marco."),
                row(1, "SPK_002", "Sure."),
            ],
        )
        self.assertEqual([(event["cluster"], event["name"]) for event in events], [("SPK_002", "Marco")])

    def test_third_person_mention_is_not_an_event(self):
        events = direct_response_events(
            "S1",
            [
                row(0, "SPK_001", "I spoke with Maya."),
                row(1, "SPK_002", "What did she say?"),
            ],
        )
        self.assertEqual(events, [])

    def test_discourse_marker_is_not_a_name(self):
        events = direct_response_events(
            "S1",
            [
                row(0, "SPK_001", "Gotcha, that makes sense."),
                row(1, "SPK_002", "Great."),
            ],
        )
        self.assertEqual(events, [])

    def test_multiple_addressees_are_rejected(self):
        events = direct_response_events(
            "S1",
            [
                row(0, "SPK_001", "Noah, Leah, can either of you answer?"),
                row(1, "SPK_002", "I can."),
            ],
        )
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
