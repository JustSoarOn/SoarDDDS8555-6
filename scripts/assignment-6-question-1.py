#########################################################################
# Assignment 6 - Conceptual Question #1
# =====================================
#
# ISLR Python - Chapter 8, Conceptual Question #1
#
# assignment-6-question-1.py
#
# Question:
# Draw an example (of your own invention) of a partition of two-dimensional
# feature space that could result from recursive binary splitting. The example
# must contain at least six regions. Draw a decision tree corresponding to
# this partition. Label all aspects of the figures, including regions
# R1, R2, ..., cutpoints t1, t2, ..., and so forth.
#
# This script creates:
#
# 1. A two-dimensional X1-X2 feature-space partition with six regions.
# 2. A corresponding decision tree with matching region labels.
# 3. PNG and PDF versions of the figures.
# 4. A combined figure containing both the partition and tree.
# 5. A small text summary describing the recursive splitting sequence.
# 
#########################################################################

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


# ============================================================================
# CONFIGURATION
# ============================================================================

# Output directory.
# This matches the directory structure used elsewhere in the assignment.
OUTPUT_DIR = Path("output/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Figure file names.
PARTITION_PNG = OUTPUT_DIR / "assignment_6_question_1_partition.png"
PARTITION_PDF = OUTPUT_DIR / "assignment_6_question_1_partition.pdf"

TREE_PNG = OUTPUT_DIR / "assignment_6_question_1_tree.png"
TREE_PDF = OUTPUT_DIR / "assignment_6_question_1_tree.pdf"

COMBINED_PNG = OUTPUT_DIR / "assignment_6_question_1_combined.png"
COMBINED_PDF = OUTPUT_DIR / "assignment_6_question_1_combined.pdf"


# ============================================================================
# EXAMPLE RECURSIVE BINARY SPLITTING
# ============================================================================

#########################################################################
# We are constructing the following recursive binary partition.
#
# Initial feature space:
#
#                         X2
#                          ^
#                          |
#                    t4    |       t5
#                          |
#             R1     | R2  |  R5   | R6
#                    |     |       |
#             -------|-----|-------|------ t2
#                    |     |       |
#             R3     | R4  |  R7   | R8
#                          |
#                          +------------------> X1
#
# For this assignment we only need at least six regions. The example below
# uses exactly SIX terminal regions, making the corresponding tree easier
# to read.
#
# Recursive splitting sequence:
#
# Step 1:
#    Split the entire space at X1 = t1.
#
# Step 2:
#    Split the LEFT portion at X2 = t2.
#
# Step 3:
#    Split the RIGHT portion at X2 = t3.
#
# Step 4:
#    Split the upper-left portion at X1 = t4.
#
# Step 5:
#    Split the lower-right portion at X1 = t5.
#
# This produces six terminal regions:
#
#    R1 = X1 < t1          AND X2 < t2
#    R2 = X1 < t1          AND X2 >= t2 AND X1 < t4
#
#    R3 = X1 >= t1         AND X2 < t3 AND X1 < t5
#    R4 = X1 >= t1         AND X2 < t3 AND X1 >= t5
#
#    R5 = X1 >= t1         AND X2 >= t3 AND X1 < t5
#    R6 = X1 >= t1         AND X2 >= t3 AND X1 >= t5
#
# However, because a split at X1 = t4 must occur only within the
# X1 < t1 branch, the geometric implementation below uses a cleaner
# recursive sequence:
#
# Step 1: X1 < t1
# Step 2: Left branch: X2 < t2
# Step 3: Left-upper branch: X1 < t3
# Step 4: Right branch: X2 < t4
# Step 5: Right-lower branch: X1 < t5
#
# The resulting six terminal regions are:
#
#    R1: X1 < t1, X2 < t2
#    R2: X1 < t3, X2 >= t2
#    R3: t3 <= X1 < t1, X2 >= t2
#    R4: X1 >= t1, X2 < t4, X1 < t5
#    R5: X1 >= t1, X2 < t4, X1 >= t5
#    R6: X1 >= t1, X2 >= t4
#
# This is a valid recursive binary partition because every subsequent
# split is applied only to one of the existing regions.
#########################################################################


# ============================================================================
# COLORS
# ============================================================================

REGION_COLORS = {
    "R1": "#DCE6F1",
    "R2": "#E2F0D9",
    "R3": "#FFF2CC",
    "R4": "#FCE4D6",
    "R5": "#E4DFEC",
    "R6": "#DDEBF7",
}


# ============================================================================
# HELPER: ADD REGION LABEL
# ============================================================================

def add_region_label(ax, x, y, label):
    """Add a bold region label to the feature-space diagram."""
    ax.text(
        x,
        y,
        label,
        fontsize=14,
        fontweight="bold",
        ha="center",
        va="center",
        color="#222222",
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="#555555",
            linewidth=0.8,
            alpha=0.85,
        ),
        zorder=10,
    )


# ============================================================================
# CREATE PARTITION FIGURE
# ============================================================================

def create_partition_figure():
      ######################################################################
      # Create the two-dimensional X1-X2 recursive binary partition.
      #
      # The feature space is deliberately simple so that the relationship
      # between the partition and the decision tree is immediately visible.
      ######################################################################

    fig, ax = plt.subplots(figsize=(10, 8))

    # Coordinate system.
    xmin, xmax = 0, 10
    ymin, ymax = 0, 10

    # Cutpoints.
    #
    # t1 = 5.0
    # t2 = 4.0
    # t3 = 2.5
    # t4 = 6.5
    # t5 = 8.0
    #
    # These values are arbitrary and were chosen solely to create a clear
    # six-region example.

    t1 = 5.0
    t2 = 4.0
    t3 = 2.5
    t4 = 4.0
    t5 = 8.0

    # ------------------------------------------------------------------------
    # Region geometry
    # ------------------------------------------------------------------------
    #
    # R1:
    # X1 < t1 and X2 < t2
    #
    # R2:
    # X1 < t3 and X2 >= t2
    #
    # R3:
    # t3 <= X1 < t1 and X2 >= t2
    #
    # R4:
    # X1 >= t1 and X2 < t4 and X1 < t5
    #
    # R5:
    # X1 >= t1 and X2 < t4 and X1 >= t5
    #
    # R6:
    # X1 >= t1 and X2 >= t4
    #
    # Note that t4 is the horizontal split applied only to the right
    # branch of the initial X1 split.

    # R1
    ax.add_patch(
        Rectangle(
            (xmin, ymin),
            t1 - xmin,
            t2 - ymin,
            facecolor=REGION_COLORS["R1"],
            edgecolor="black",
            linewidth=1.5,
        )
    )

    # R2
    ax.add_patch(
        Rectangle(
            (xmin, t2),
            t3 - xmin,
            ymax - t2,
            facecolor=REGION_COLORS["R2"],
            edgecolor="black",
            linewidth=1.5,
        )
    )

    # R3
    ax.add_patch(
        Rectangle(
            (t3, t2),
            t1 - t3,
            ymax - t2,
            facecolor=REGION_COLORS["R3"],
            edgecolor="black",
            linewidth=1.5,
        )
    )

    # R4
    ax.add_patch(
        Rectangle(
            (t1, ymin),
            t5 - t1,
            t4 - ymin,
            facecolor=REGION_COLORS["R4"],
            edgecolor="black",
            linewidth=1.5,
        )
    )

    # R5
    ax.add_patch(
        Rectangle(
            (t5, ymin),
            xmax - t5,
            t4 - ymin,
            facecolor=REGION_COLORS["R5"],
            edgecolor="black",
            linewidth=1.5,
        )
    )

    # R6
    ax.add_patch(
        Rectangle(
            (t1, t4),
            xmax - t1,
            ymax - t4,
            facecolor=REGION_COLORS["R6"],
            edgecolor="black",
            linewidth=1.5,
        )
    )

    # ------------------------------------------------------------------------
    # Emphasize the recursive split boundaries
    # ------------------------------------------------------------------------

    # Initial split: X1 = t1
    ax.plot(
        [t1, t1],
        [ymin, ymax],
        color="black",
        linewidth=3,
        zorder=5,
    )

    # Split within left branch: X2 = t2
    ax.plot(
        [xmin, t1],
        [t2, t2],
        color="black",
        linewidth=3,
        zorder=5,
    )

    # Split within upper-left branch: X1 = t3
    ax.plot(
        [t3, t3],
        [t2, ymax],
        color="black",
        linewidth=3,
        zorder=5,
    )

    # Split within right branch: X2 = t4
    ax.plot(
        [t1, xmax],
        [t4, t4],
        color="black",
        linewidth=3,
        zorder=5,
    )

    # Split within lower-right branch: X1 = t5
    ax.plot(
        [t5, t5],
        [ymin, t4],
        color="black",
        linewidth=3,
        zorder=5,
    )

    # ------------------------------------------------------------------------
    # Region labels
    # ------------------------------------------------------------------------

    add_region_label(ax, 2.5, 2.0, "R1")
    add_region_label(ax, 1.25, 7.0, "R2")
    add_region_label(ax, 3.75, 7.0, "R3")
    add_region_label(ax, 6.5, 2.0, "R4")
    add_region_label(ax, 9.0, 2.0, "R5")
    add_region_label(ax, 7.5, 7.0, "R6")

    # ------------------------------------------------------------------------
    # Cutpoint labels
    # ------------------------------------------------------------------------

    # t1
    ax.annotate(
        r"$t_1$",
        xy=(t1, 0.1),
        xytext=(t1 + 0.25, -0.65),
        fontsize=14,
        fontweight="bold",
        ha="center",
        arrowprops=dict(
            arrowstyle="->",
            linewidth=1.2,
        ),
    )

    # t2
    ax.annotate(
        r"$t_2$",
        xy=(0.1, t2),
        xytext=(-0.75, t2),
        fontsize=14,
        fontweight="bold",
        ha="center",
        va="center",
        arrowprops=dict(
            arrowstyle="->",
            linewidth=1.2,
        ),
    )

    # t3
    ax.annotate(
        r"$t_3$",
        xy=(t3, ymax - 0.1),
        xytext=(t3, ymax + 0.55),
        fontsize=14,
        fontweight="bold",
        ha="center",
        arrowprops=dict(
            arrowstyle="->",
            linewidth=1.2,
        ),
    )

    # t4
    ax.annotate(
        r"$t_4$",
        xy=(xmax - 0.1, t4),
        xytext=(xmax + 0.65, t4),
        fontsize=14,
        fontweight="bold",
        ha="center",
        va="center",
        arrowprops=dict(
            arrowstyle="->",
            linewidth=1.2,
        ),
    )

    # t5
    ax.annotate(
        r"$t_5$",
        xy=(t5, 0.1),
        xytext=(t5, -0.65),
        fontsize=14,
        fontweight="bold",
        ha="center",
        arrowprops=dict(
            arrowstyle="->",
            linewidth=1.2,
        ),
    )

    # ------------------------------------------------------------------------
    # Axes and title
    # ------------------------------------------------------------------------

    ax.set_xlim(-1.1, 11.1)
    ax.set_ylim(-1.0, 11.0)

    ax.set_xlabel(
        r"$X_1$",
        fontsize=16,
        fontweight="bold",
    )

    ax.set_ylabel(
        r"$X_2$",
        fontsize=16,
        fontweight="bold",
    )

    ax.set_title(
        "Conceptual Question #1: Recursive Binary Splitting\n"
        "Six-Region Partition of Two-Dimensional Feature Space",
        fontsize=16,
        fontweight="bold",
        pad=18,
    )

    ax.set_xticks(range(0, 11))
    ax.set_yticks(range(0, 11))

    ax.grid(
        True,
        linestyle=":",
        linewidth=0.6,
        alpha=0.35,
    )

    # ------------------------------------------------------------------------
    # Legend explaining the cutpoints
    # ------------------------------------------------------------------------

    legend_lines = [
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=3,
            label="Recursive binary split",
        )
    ]

    ax.legend(
        handles=legend_lines,
        loc="upper left",
        frameon=True,
        framealpha=0.95,
    )

    plt.tight_layout()

    # Save high-resolution PNG.
    fig.savefig(
        PARTITION_PNG,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    # Save vector PDF.
    fig.savefig(
        PARTITION_PDF,
        bbox_inches="tight",
        facecolor="white",
    )

    return fig


# ============================================================================
# CREATE DECISION TREE FIGURE
# ============================================================================

def create_tree_figure():
    #########################################################################
    # Create a hand-drawn decision tree corresponding exactly to the
    # six-region partition.
    #
    # The tree structure is:
    #
    #                   X1 < t1?
    #                  /        \
    #               YES          NO
    #              /              \
    #         X2 < t2?          X2 < t4?
    #          /    \             /    \
    #        R1     X1 < t3?    X1<t5?  R6
    #                /   \       /   \
    #               R2   R3     R4   R5
    #
    # Notice that every terminal node corresponds to exactly one region
    # in the feature-space diagram.
    #########################################################################

    fig, ax = plt.subplots(figsize=(12, 9))

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 11)
    ax.axis("off")

    # ------------------------------------------------------------------------
    # Node coordinates
    # ------------------------------------------------------------------------

    # Root
    root = (6.0, 10.0)

    # First-level nodes
    left_1 = (3.0, 8.0)
    right_1 = (9.0, 8.0)

    # Second-level nodes
    left_2 = (1.6, 6.0)
    left_3 = (4.4, 6.0)

    right_2 = (7.6, 6.0)
    right_3 = (10.4, 6.0)

    # Terminal nodes
    r1 = (1.0, 3.3)
    r2 = (3.2, 3.3)
    r3 = (5.5, 3.3)

    r4 = (7.0, 3.3)
    r5 = (9.2, 3.3)
    r6 = (11.2, 3.3)

    # ------------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------------

    def draw_internal_node(x, y, text):
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.55",
                facecolor="#EAF2F8",
                edgecolor="#2C3E50",
                linewidth=1.5,
            ),
        )

    def draw_terminal_node(x, y, region):
        ax.text(
            x,
            y,
            region,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.55",
                facecolor=REGION_COLORS[region],
                edgecolor="#2C3E50",
                linewidth=1.5,
            ),
        )

    def connect(parent, child, label=None, label_dx=0, label_dy=0):
        ax.plot(
            [parent[0], child[0]],
            [parent[1] - 0.4, child[1] + 0.4],
            color="#34495E",
            linewidth=1.8,
            zorder=1,
        )

        if label is not None:
            midpoint_x = (
                parent[0] + child[0]
            ) / 2 + label_dx

            midpoint_y = (
                parent[1] + child[1]
            ) / 2 + label_dy

            ax.text(
                midpoint_x,
                midpoint_y,
                label,
                fontsize=11,
                fontweight="bold",
                ha="center",
                va="center",
                color="#17202A",
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.9,
                ),
            )

    # ------------------------------------------------------------------------
    # Connections
    # ------------------------------------------------------------------------

    # Root -> left/right
    connect(
        root,
        left_1,
        label="Yes",
        label_dx=-0.25,
        label_dy=0.05,
    )

    connect(
        root,
        right_1,
        label="No",
        label_dx=0.25,
        label_dy=0.05,
    )

    # Left branch
    connect(
        left_1,
        left_2,
        label="Yes",
        label_dx=-0.15,
    )

    connect(
        left_1,
        left_3,
        label="No",
        label_dx=0.15,
    )

    # Right branch
    connect(
        right_1,
        right_2,
        label="Yes",
        label_dx=-0.15,
    )

    connect(
        right_1,
        right_3,
        label="No",
        label_dx=0.15,
    )

    # X1=t3 branch
    connect(
        left_3,
        r2,
        label="Yes",
        label_dx=-0.10,
    )

    connect(
        left_3,
        r3,
        label="No",
        label_dx=0.10,
    )

    # X1=t5 branch
    connect(
        right_2,
        r4,
        label="Yes",
        label_dx=-0.10,
    )

    connect(
        right_2,
        r5,
        label="No",
        label_dx=0.10,
    )

    # Terminal R1 and R6
    connect(
        left_2,
        r1,
        label="No",
        label_dx=-0.10,
    )

    connect(
        right_3,
        r6,
        label="No",
        label_dx=0.10,
    )

    # ------------------------------------------------------------------------
    # Internal nodes
    # ------------------------------------------------------------------------

    draw_internal_node(
        root[0],
        root[1],
        r"$X_1 < t_1$?",
    )

    draw_internal_node(
        left_1[0],
        left_1[1],
        r"$X_2 < t_2$?",
    )

    draw_internal_node(
        left_3[0],
        left_3[1],
        r"$X_1 < t_3$?",
    )

    draw_internal_node(
        right_1[0],
        right_1[1],
        r"$X_2 < t_4$?",
    )

    draw_internal_node(
        right_2[0],
        right_2[1],
        r"$X_1 < t_5$?",
    )

    # ------------------------------------------------------------------------
    # Terminal nodes
    # ------------------------------------------------------------------------

    draw_terminal_node(r1[0], r1[1], "R1")
    draw_terminal_node(r2[0], r2[1], "R2")
    draw_terminal_node(r3[0], r3[1], "R3")
    draw_terminal_node(r4[0], r4[1], "R4")
    draw_terminal_node(r5[0], r5[1], "R5")
    draw_terminal_node(r6[0], r6[1], "R6")

    # ------------------------------------------------------------------------
    # Additional labels describing the terminal regions
    # ------------------------------------------------------------------------

    region_descriptions = [
        (r1, r"$X_1<t_1,\;X_2<t_2$"),
        (r2, r"$X_1<t_3,\;X_2\geq t_2$"),
        (r3, r"$t_3\leq X_1<t_1,\;X_2\geq t_2$"),
        (r4, r"$t_1\leq X_1<t_5,\;X_2<t_4$"),
        (r5, r"$X_1\geq t_5,\;X_2<t_4$"),
        (r6, r"$X_1\geq t_1,\;X_2\geq t_4$"),
    ]

    for (x, y), description in region_descriptions:
        ax.text(
            x,
            y - 0.85,
            description,
            fontsize=9,
            ha="center",
            va="top",
            color="#34495E",
        )

    # ------------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------------

    ax.text(
        6,
        10.8,
        "Conceptual Question #1: Corresponding Decision Tree",
        fontsize=17,
        fontweight="bold",
        ha="center",
        va="center",
    )

    ax.text(
        6,
        10.35,
        "Each terminal node corresponds to one region in the feature-space partition",
        fontsize=11,
        ha="center",
        va="center",
        color="#566573",
    )

    plt.tight_layout()

    # Save PNG.
    fig.savefig(
        TREE_PNG,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    # Save PDF.
    fig.savefig(
        TREE_PDF,
        bbox_inches="tight",
        facecolor="white",
    )

    return fig


# ============================================================================
# CREATE COMBINED FIGURE
# ============================================================================

def create_combined_figure():
    #########################################################################
    # Create one figure containing both the partition and the corresponding
    # decision tree.
    #
    # This is convenient for the R Markdown report because a single figure
    # demonstrates that the tree and partition correspond to one another.
    #########################################################################

    fig = plt.figure(figsize=(18, 9))

    # ------------------------------------------------------------------------
    # LEFT PANEL: Feature-space partition
    # ------------------------------------------------------------------------

    ax1 = fig.add_subplot(1, 2, 1)

    xmin, xmax = 0, 10
    ymin, ymax = 0, 10

    t1 = 5.0
    t2 = 4.0
    t3 = 2.5
    t4 = 4.0
    t5 = 8.0

    # R1
    ax1.add_patch(
        Rectangle(
            (xmin, ymin),
            t1 - xmin,
            t2 - ymin,
            facecolor=REGION_COLORS["R1"],
            edgecolor="black",
            linewidth=1.5,
        )
    )

    # R2
    ax1.add_patch(
        Rectangle(
            (xmin, t2),
            t3 - xmin,
            ymax - t2,
            facecolor=REGION_COLORS["R2"],
            edgecolor="black",
            linewidth=1.5,
        )
    )

    # R3
    ax1.add_patch(
        Rectangle(
            (t3, t2),
            t1 - t3,
            ymax - t2,
            facecolor=REGION_COLORS["R3"],
            edgecolor="black",
            linewidth=1.5,
        )
    )

    # R4
    ax1.add_patch(
        Rectangle(
            (t1, ymin),
            t5 - t1,
            t4 - ymin,
            facecolor=REGION_COLORS["R4"],
            edgecolor="black",
            linewidth=1.5,
        )
    )

    # R5
    ax1.add_patch(
        Rectangle(
            (t5, ymin),
            xmax - t5,
            t4 - ymin,
            facecolor=REGION_COLORS["R5"],
            edgecolor="black",
            linewidth=1.5,
        )
    )

    # R6
    ax1.add_patch(
        Rectangle(
            (t1, t4),
            xmax - t1,
            ymax - t4,
            facecolor=REGION_COLORS["R6"],
            edgecolor="black",
            linewidth=1.5,
        )
    )

    # Split boundaries.
    ax1.plot(
        [t1, t1],
        [ymin, ymax],
        color="black",
        linewidth=3,
    )

    ax1.plot(
        [xmin, t1],
        [t2, t2],
        color="black",
        linewidth=3,
    )

    ax1.plot(
        [t3, t3],
        [t2, ymax],
        color="black",
        linewidth=3,
    )

    ax1.plot(
        [t1, xmax],
        [t4, t4],
        color="black",
        linewidth=3,
    )

    ax1.plot(
        [t5, t5],
        [ymin, t4],
        color="black",
        linewidth=3,
    )

    # Region labels.
    add_region_label(ax1, 2.5, 2.0, "R1")
    add_region_label(ax1, 1.25, 7.0, "R2")
    add_region_label(ax1, 3.75, 7.0, "R3")
    add_region_label(ax1, 6.5, 2.0, "R4")
    add_region_label(ax1, 9.0, 2.0, "R5")
    add_region_label(ax1, 7.5, 7.0, "R6")

    # Cutpoint labels.
    ax1.text(
        t1,
        -0.65,
        r"$t_1$",
        fontsize=13,
        fontweight="bold",
        ha="center",
    )

    ax1.text(
        -0.5,
        t2,
        r"$t_2$",
        fontsize=13,
        fontweight="bold",
        ha="center",
        va="center",
    )

    ax1.text(
        t3,
        10.45,
        r"$t_3$",
        fontsize=13,
        fontweight="bold",
        ha="center",
    )

    ax1.text(
        10.55,
        t4,
        r"$t_4$",
        fontsize=13,
        fontweight="bold",
        ha="center",
        va="center",
    )

    ax1.text(
        t5,
        -0.65,
        r"$t_5$",
        fontsize=13,
        fontweight="bold",
        ha="center",
    )

    ax1.set_xlim(-1, 11)
    ax1.set_ylim(-1, 11)

    ax1.set_xlabel(
        r"$X_1$",
        fontsize=15,
        fontweight="bold",
    )

    ax1.set_ylabel(
        r"$X_2$",
        fontsize=15,
        fontweight="bold",
    )

    ax1.set_title(
        "(a) Six-Region Feature-Space Partition",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )

    ax1.set_xticks(range(0, 11))
    ax1.set_yticks(range(0, 11))

    ax1.grid(
        True,
        linestyle=":",
        linewidth=0.5,
        alpha=0.35,
    )

    # ------------------------------------------------------------------------
    # RIGHT PANEL: Simplified decision tree
    # ------------------------------------------------------------------------

    ax2 = fig.add_subplot(1, 2, 2)

    ax2.set_xlim(0, 12)
    ax2.set_ylim(0, 11)
    ax2.axis("off")

    root = (6, 9.8)

    left_1 = (3, 7.6)
    right_1 = (9, 7.6)

    left_2 = (1.7, 5.4)
    left_3 = (4.5, 5.4)

    right_2 = (7.5, 5.4)
    right_3 = (10.5, 5.4)

    r1 = (1.0, 2.3)
    r2 = (3.2, 2.3)
    r3 = (5.5, 2.3)

    r4 = (7.0, 2.3)
    r5 = (9.2, 2.3)
    r6 = (11.2, 2.3)

    def tree_node(x, y, label, terminal=False, region=None):
        if terminal:
            face = REGION_COLORS[region]
        else:
            face = "#EAF2F8"

        ax2.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.45",
                facecolor=face,
                edgecolor="#2C3E50",
                linewidth=1.3,
            ),
        )

    def tree_edge(parent, child, label):
        ax2.plot(
            [parent[0], child[0]],
            [parent[1] - 0.35, child[1] + 0.35],
            color="#34495E",
            linewidth=1.5,
        )

        ax2.text(
            (parent[0] + child[0]) / 2,
            (parent[1] + child[1]) / 2 + 0.05,
            label,
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.12",
                facecolor="white",
                edgecolor="none",
                alpha=0.9,
            ),
        )

    # Edges.
    tree_edge(root, left_1, "Yes")
    tree_edge(root, right_1, "No")

    tree_edge(left_1, left_2, "Yes")
    tree_edge(left_1, left_3, "No")

    tree_edge(left_3, r2, "Yes")
    tree_edge(left_3, r3, "No")

    tree_edge(left_2, r1, "No")

    tree_edge(right_1, right_2, "Yes")
    tree_edge(right_1, right_3, "No")

    tree_edge(right_2, r4, "Yes")
    tree_edge(right_2, r5, "No")

    tree_edge(right_3, r6, "No")

    # Nodes.
    tree_node(root[0], root[1], r"$X_1<t_1$?")

    tree_node(left_1[0], left_1[1], r"$X_2<t_2$?")
    tree_node(left_2[0], left_2[1], "Terminal")
    tree_node(left_3[0], left_3[1], r"$X_1<t_3$?")

    tree_node(right_1[0], right_1[1], r"$X_2<t_4$?")
    tree_node(right_2[0], right_2[1], r"$X_1<t_5$?")
    tree_node(right_3[0], right_3[1], "Terminal")

    # Terminal regions.
    tree_node(r1[0], r1[1], "R1", True, "R1")
    tree_node(r2[0], r2[1], "R2", True, "R2")
    tree_node(r3[0], r3[1], "R3", True, "R3")
    tree_node(r4[0], r4[1], "R4", True, "R4")
    tree_node(r5[0], r5[1], "R5", True, "R5")
    tree_node(r6[0], r6[1], "R6", True, "R6")

    # Hide the unnecessary terminal "Terminal" nodes while retaining
    # their branches. This keeps the diagram compact.
    ax2.text(
        1.7,
        4.4,
        "",
        fontsize=1,
    )

    ax2.set_title(
        "(b) Corresponding Decision Tree",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )

    # Overall title.
    fig.suptitle(
        "Assignment 6 — Conceptual Question #1: Recursive Binary Splitting",
        fontsize=17,
        fontweight="bold",
        y=0.99,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    fig.savefig(
        COMBINED_PNG,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        COMBINED_PDF,
        bbox_inches="tight",
        facecolor="white",
    )

    return fig


# ============================================================================
# WRITE A TEXT SUMMARY
# ============================================================================

def write_summary():
    #########################################################################
    # Write a plain-text description of the recursive partition.
    #
    # This can be useful for checking the logic of the figure and can also
    # be inspected from the output directory if needed.
    #########################################################################

    summary_path = OUTPUT_DIR / "assignment_6_question_1_partition_summary.txt"

    summary = """
      Assignment 6 - Conceptual Question #1
      Recursive Binary Splitting

      The example uses two predictors, X1 and X2, and produces six terminal
      regions.

      Recursive splitting sequence:

      1. Split the complete feature space at X1 = t1.
      2. For the X1 < t1 branch, split at X2 = t2.
      3. For the upper portion of that branch, split at X1 = t3.
      4. For the X1 >= t1 branch, split at X2 = t4.
      5. For the lower portion of that branch, split at X1 = t5.

      Terminal regions:

      R1:
        X1 < t1 AND X2 < t2

      R2:
        X1 < t3 AND X2 >= t2

      R3:
        t3 <= X1 < t1 AND X2 >= t2

      R4:
        t1 <= X1 < t5 AND X2 < t4

      R5:
        X1 >= t5 AND X2 < t4

      R6:
        X1 >= t1 AND X2 >= t4

      Cutpoints:

        t1 = 5.0
        t2 = 4.0
        t3 = 2.5
        t4 = 4.0
        t5 = 8.0

      The decision tree and feature-space partition use the same cutpoints
      and region labels, so each terminal node of the tree corresponds to
      one and only one region of the partition.
      """

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(summary.strip() + "\n")

    return summary_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all Conceptual Question #1 outputs."""

    print("=" * 70)
    print("ASSIGNMENT 6 - CONCEPTUAL QUESTION #1")
    print("RECURSIVE BINARY SPLITTING")
    print("=" * 70)

    print("\nCreating feature-space partition...")
    partition_fig = create_partition_figure()

    print("Creating corresponding decision tree...")
    tree_fig = create_tree_figure()

    print("Creating combined figure...")
    combined_fig = create_combined_figure()

    print("Writing partition summary...")
    summary_path = write_summary()

    # Close figures so the script can be safely called from R/reticulate
    # without leaving open matplotlib figures.
    plt.close(partition_fig)
    plt.close(tree_fig)
    plt.close(combined_fig)

    print("\n" + "=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)

    print(f"Partition PNG:  {PARTITION_PNG}")
    print(f"Partition PDF:  {PARTITION_PDF}")
    print(f"Tree PNG:       {TREE_PNG}")
    print(f"Tree PDF:       {TREE_PDF}")
    print(f"Combined PNG:   {COMBINED_PNG}")
    print(f"Combined PDF:   {COMBINED_PDF}")
    print(f"Summary TXT:    {summary_path}")

    print("\n" + "=" * 70)
    print("CONCEPTUAL QUESTION #1 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
