"""Iris Pulse Network: pulse propagation + single-pulse plasticity.

PCA is not used. The four original Iris features are converted into pulses.
A pulse travels Input -> Memory -> Class. An isolated pulse changes the
outgoing links of its memory node (short-term synaptic plasticity).
"""

import numpy as np
import turtle

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ------------------------------------------------------------
# 1. Iris data: use the original four dimensions (no PCA)
# ------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

iris = load_iris()
X, y = iris.data, iris.target
class_names = iris.target_names
feature_names = ["がく片の長さ", "がく片の幅", "花弁の長さ", "花弁の幅"]
N_FEATURES = X.shape[1]
N_CLASSES = len(class_names)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=SEED, stratify=y
)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)


# ------------------------------------------------------------
# 2. Pulse settings
# ------------------------------------------------------------
MAX_PULSES = 12
SINGLE_THRESHOLD = 0.85
K_NEIGHBORS = 5

TRACE_DECAY = 0.55       # old single-pulse traces fade between samples
SINGLE_BOOST = 0.75      # strength added by one isolated pulse
PLASTICITY_GAIN = 1.25   # effect of the trace on memory -> class links


def encode_pulses(x):
    """Return periodic pulse rates and isolated-pulse flags."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    counts = 1 + np.rint(x * (MAX_PULSES - 1)).astype(int)
    periodic = counts / MAX_PULSES
    single = (x >= SINGLE_THRESHOLD).astype(float)
    return periodic, single


def pulse_code(x):
    """Compact pulse pattern used for the memory-based classifier."""
    periodic, single = encode_pulses(x)
    return np.r_[periodic, 0.5 * single]


# ------------------------------------------------------------
# 3. Build the pulse memory network
#    Input nodes -> Memory nodes -> Class nodes
# ------------------------------------------------------------
class PulseNetwork:
    def __init__(self, x_train, labels):
        self.labels = np.asarray(labels)
        encoded = [encode_pulses(x) for x in x_train]
        self.periodic = np.asarray([p[0] for p in encoded])
        self.single = np.asarray([p[1] for p in encoded])
        self.memory_codes = np.asarray([pulse_code(x) for x in x_train])

        # Learned links from memory nodes to class nodes.
        self.base_links = self._learn_periodic_links()
        self.single_links = self._learn_single_links()

        # Each memory node has a short-lived plasticity trace.
        self.trace = np.zeros(N_FEATURES)

    def _learn_periodic_links(self):
        class_means = np.vstack([
            self.periodic[self.labels == c].mean(axis=0)
            for c in range(N_CLASSES)
        ])
        links = (class_means - self.periodic.mean(axis=0)).T
        scale = np.max(np.abs(links), axis=1, keepdims=True)
        return links / np.maximum(scale, 1e-8)

    def _learn_single_links(self):
        prior = np.bincount(self.labels, minlength=N_CLASSES).astype(float)
        prior /= prior.sum()
        links = np.zeros((N_FEATURES, N_CLASSES))

        for f in range(N_FEATURES):
            counts = np.array([
                np.sum((self.single[:, f] > 0) & (self.labels == c))
                for c in range(N_CLASSES)
            ], dtype=float)
            if counts.sum() == 0:
                continue

            # Compare P(class | isolated pulse) with the ordinary class prior.
            posterior = (counts + 1.0) / (counts.sum() + N_CLASSES)
            row = np.log(posterior / prior)
            row -= row.mean()
            scale = np.max(np.abs(row))
            if scale > 1e-8:
                row /= scale
            links[f] = row
        return links

    def begin_sample(self):
        """Let old short-term traces fade before the next input pattern."""
        self.trace *= TRACE_DECAY

    def receive_single_pulse(self, feature_id):
        """A single pulse physically changes the effective outgoing links."""
        self.trace[feature_id] = min(1.0, self.trace[feature_id] + SINGLE_BOOST)

    def links_now(self):
        links = self.base_links + (
            PLASTICITY_GAIN * self.trace[:, None] * self.single_links
        )
        return np.clip(links, -2.0, 2.0)

    def predict(self, x):
        """Use the five closest stored pulse patterns for classification."""
        code = pulse_code(x)
        distances = np.linalg.norm(self.memory_codes - code, axis=1)
        nearest = np.argsort(distances)[:K_NEIGHBORS]
        labels, counts = np.unique(self.labels[nearest], return_counts=True)
        return int(labels[np.argmax(counts)])


network = PulseNetwork(X_train, y_train)


# ------------------------------------------------------------
# 4. Evaluate the classifier
# ------------------------------------------------------------
y_pred = np.array([network.predict(x) for x in X_test])

print("=== Iris Pulse Network ===")
print("PCA: not used; all four original features are used.")
print(f"Maximum periodic pulses : {MAX_PULSES}")
print(f"Single-pulse threshold  : {SINGLE_THRESHOLD:.2f}")
print(f"Nearest memories        : {K_NEIGHBORS}")
print(f"Test accuracy           : {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification report:")
print(classification_report(
    y_test, y_pred, target_names=class_names, zero_division=0
))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))


# ------------------------------------------------------------
# 5. Create events for the Turtle animation
# ------------------------------------------------------------
def make_events(x):
    """Each event is (time, feature_id, 'periodic' or 'single')."""
    events = []
    for f, value in enumerate(np.clip(x, 0.0, 1.0)):
        count = 1 + int(round(value * (MAX_PULSES - 1)))
        for t in np.linspace(0.05, 0.72, count):
            events.append((float(t), f, "periodic"))
        if value >= SINGLE_THRESHOLD:
            events.append((0.88, f, "single"))
    return sorted(events)


# ------------------------------------------------------------
# 6. Turtle: animate Input -> Memory -> Class pulse propagation
# ------------------------------------------------------------
def show_turtle(x, true_class, predicted_class, model):
    events = make_events(x)

    screen = turtle.Screen()
    screen.setup(1300, 900)
    screen.title("Iris Pulse Network - Propagation and Plasticity")
    screen.bgcolor("white")
    screen.tracer(0, 0)

    input_xy = [(-470, 205), (-470, 80), (-470, -45), (-470, -170)]
    memory_xy = [(-60, 205), (-60, 80), (-60, -45), (-60, -170)]
    class_xy = [(405, 145), (405, 0), (405, -145)]
    feature_colors = ["#2878B5", "#E07A24", "#299D65", "#9B59B6"]

    graph = turtle.Turtle(visible=False)
    graph.speed(0)
    ui = turtle.Turtle(visible=False)
    ui.speed(0)
    timeline = turtle.Turtle(visible=False)
    timeline.speed(0)
    cursor = turtle.Turtle(visible=False)
    cursor.speed(0)
    pulse = turtle.Turtle(shape="circle")
    pulse.speed(0)
    pulse.penup()
    pulse.shapesize(0.8, 0.8)
    pulse.hideturtle()

    def line(pen, a, b, color="#999999", width=1):
        pen.penup()
        pen.goto(a)
        pen.color(color)
        pen.pensize(width)
        pen.pendown()
        pen.goto(b)
        pen.penup()

    def label(pen, pos, message, color="black", size=10, align="left"):
        pen.penup()
        pen.goto(pos)
        pen.color(color)
        pen.write(message, align=align, font=("Arial", size, "normal"))

    def node(pen, pos, name, color, radius=23):
        pen.penup()
        pen.goto(pos[0], pos[1] - radius)
        pen.color(color, "white")
        pen.pensize(2)
        pen.pendown()
        pen.begin_fill()
        pen.circle(radius)
        pen.end_fill()
        pen.penup()
        label(pen, (pos[0], pos[1] - 5), name, color, 10, "center")

    def draw_graph():
        graph.clear()
        current = model.links_now()

        # Fixed links: an input pulse always triggers its matching memory node.
        for f in range(N_FEATURES):
            line(graph, (input_xy[f][0] + 24, input_xy[f][1]),
                 (memory_xy[f][0] - 24, memory_xy[f][1]), "#8B929A", 2)

        # Adjustable links: each memory node can trigger a class node.
        for f in range(N_FEATURES):
            for c in range(N_CLASSES):
                w = current[f, c]
                if model.trace[f] > 0.05:
                    color = "#D69E22"  # gold = changed by a single pulse
                    width = 2 + min(3, abs(w) * 2)
                elif w >= 0:
                    color = "#438BC4"
                    width = 1 + min(3, abs(w) * 2)
                else:
                    color = "#D16B67"
                    width = 1 + min(3, abs(w) * 2)

                a = (memory_xy[f][0] + 24, memory_xy[f][1])
                b = (class_xy[c][0] - 28, class_xy[c][1])
                line(graph, a, b, color, width)
                mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 5)
                label(graph, mid, f"{w:+.2f}", color, 8, "center")

        # Draw nodes on top of the connections.
        for f in range(N_FEATURES):
            node(graph, input_xy[f], f"F{f + 1}", feature_colors[f])
            node(graph, memory_xy[f], f"M{f + 1}",
                 "#D69E22" if model.trace[f] > 0.05 else "#596B7A")
            if model.trace[f] > 0.05:
                label(graph, (memory_xy[f][0], memory_xy[f][1] + 31),
                      f"trace {model.trace[f]:.2f}", "#A87500", 8, "center")

        for c in range(N_CLASSES):
            node(graph, class_xy[c], f"C{c + 1}", "#384B5E", 27)
            label(graph, (class_xy[c][0] + 42, class_xy[c][1] - 5),
                  class_names[c], "#202833", 11)

    def draw_timeline():
        left, right = -400, 400
        ys = [-280, -315, -350, -385]
        timeline.clear()
        for f, y0 in enumerate(ys):
            line(timeline, (left, y0), (right, y0), "#BEC5CC", 1)
            label(ui, (-590, y0 - 5), feature_names[f], feature_colors[f], 8)
        for t, f, kind in events:
            xx = left + (right - left) * t
            height = 12 if kind == "periodic" else 24
            color = feature_colors[f] if kind == "periodic" else "#D69E22"
            line(timeline, (xx, ys[f]), (xx, ys[f] + height), color,
                 2 if kind == "periodic" else 4)
        label(ui, (left, -412), "0.0", size=8)
        label(ui, (right, -412), "1.0   time →", size=8, align="right")

    label(ui, (-590, 390), "IRIS PULSE NETWORK", size=19)
    label(ui, (-590, 358),
          f"True: {class_names[true_class]}     kNN prediction: {class_names[predicted_class]}",
          size=12)
    label(ui, (-590, 328),
          "A pulse travels Input → Memory → Class. A gold single pulse changes the memory-to-class links.",
          size=10)
    label(ui, (-470, 265), "INPUT", "#2878B5", 10)
    label(ui, (-60, 265), "MEMORY", "#596B7A", 10)
    label(ui, (405, 205), "OUTPUT", "#384B5E", 10, "center")

    draw_graph()
    draw_timeline()

    status = turtle.Turtle(visible=False)
    status.speed(0)

    def set_status(message, color="#273746"):
        status.clear()
        label(status, (-590, -438), message, color, 10)

    set_status("Ready: each input pulse will trigger a new pulse in its memory node.")
    screen.update()

    # Animation state: each event travels over two links.
    state = {"event": 0, "stage": 0, "step": 0, "route": None,
             "feature": None, "kind": None, "target": None, "weight": 0.0}
    STEPS = 9

    def next_event():
        if state["event"] >= len(events):
            pulse.hideturtle()
            set_status("Finished. Gold nodes and links show the network changes caused by isolated pulses.",
                       "#A87500")
            screen.update()
            return

        t, f, kind = events[state["event"]]
        state["feature"], state["kind"] = f, kind
        state["stage"], state["step"] = 0, 0
        state["route"] = [input_xy[f], memory_xy[f], None]

        # Move the time cursor to this event.
        cursor.clear()
        xx = -400 + 800 * t
        line(cursor, (xx, -270), (xx, -395), "#222222", 2)

        color = "#D69E22" if kind == "single" else feature_colors[f]
        pulse.color(color)
        pulse.goto(input_xy[f])
        pulse.showturtle()
        set_status(f"{kind.upper()} pulse at F{f + 1}: travelling to memory node M{f + 1}.")
        screen.update()
        animate_segment()

    def animate_segment():
        f = state["feature"]
        stage = state["stage"]
        step = state["step"]
        a = state["route"][stage]
        b = state["route"][stage + 1]
        ratio = step / STEPS
        pulse.goto(a[0] + (b[0] - a[0]) * ratio,
                   a[1] + (b[1] - a[1]) * ratio)
        screen.update()

        if step < STEPS:
            state["step"] += 1
            screen.ontimer(animate_segment, 22)
            return

        if stage == 0:
            # The memory node has received the first pulse and emits a new one.
            if state["kind"] == "single":
                model.receive_single_pulse(f)
                draw_graph()  # visually update the network links
                set_status(f"SINGLE pulse reached M{f + 1}: its outgoing memory links changed.",
                           "#A87500")
            else:
                set_status(f"M{f + 1} received the pulse and generated a new propagation pulse.")

            # The memory pulse follows its strongest current outgoing link.
            weights = model.links_now()[f]
            target = int(np.argmax(np.abs(weights)))
            state["target"] = target
            state["weight"] = float(weights[target])
            state["route"][2] = class_xy[target]
            state["stage"], state["step"] = 1, 0
            screen.ontimer(animate_segment, 180)
        else:
            target = state["target"]
            pulse.hideturtle()
            set_status(f"Output pulse reached {class_names[target]} (link {state['weight']:+.2f}).")
            state["event"] += 1
            screen.ontimer(next_event, 110)

    screen.ontimer(next_event, 500)
    screen.mainloop()


# ------------------------------------------------------------
# 7. Show one test sample that contains an isolated pulse
# ------------------------------------------------------------
single_ids = [i for i, x in enumerate(X_test)
              if np.any(encode_pulses(x)[1] > 0)]
demo_id = single_ids[0] if single_ids else 0

demo_x = X_test[demo_id]
true_class = int(y_test[demo_id])
predicted_class = network.predict(demo_x)

print("\nTurtle demonstration:")
print("  True class      :", class_names[true_class])
print("  Predicted class :", class_names[predicted_class])
print("  Normalized input:", np.round(demo_x, 3))
print("  Single-pulse features:", [
    feature_names[f] for f, flag in enumerate(encode_pulses(demo_x)[1]) if flag
])

network.trace[:] = 0.0
show_turtle(demo_x, true_class, predicted_class, network)
