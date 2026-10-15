"""Exact Gen-III capture arithmetic and risk-aware training/capture plans.

Integer order follows ``Cmd_handleballthrow`` in the original games: first
truncate catchRate * ballMultiplier / 10, then apply HP and status modifiers.
No catch probability is rounded to one or padded to a positive value. The
probability assumes independent uniform sixteen-bit shake draws; native RNG
correlations belong to the emulator, rather than a guarantee by this planner.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from fractions import Fraction
import math
from types import MappingProxyType


# Generated from the ROM-internal SPECIES_* IDs in pret/pokeemerald at
# 731ad5bfd6e6f265508d0efcca0ba42f9dcf5881. IDs 252..276 are old Unown
# placeholders; Hoenn starters begin at Treecko=277, Torchic=280, Mudkip=283.
SPECIES_CATCH_RATES: dict[int, int] = {
    1: 45, 2: 45, 3: 45, 4: 45, 5: 45, 6: 45, 7: 45, 8: 45,
    9: 45, 10: 255, 11: 120, 12: 45, 13: 255, 14: 120, 15: 45, 16: 255,
    17: 120, 18: 45, 19: 255, 20: 127, 21: 255, 22: 90, 23: 255, 24: 90,
    25: 190, 26: 75, 27: 255, 28: 90, 29: 235, 30: 120, 31: 45, 32: 235,
    33: 120, 34: 45, 35: 150, 36: 25, 37: 190, 38: 75, 39: 170, 40: 50,
    41: 255, 42: 90, 43: 255, 44: 120, 45: 45, 46: 190, 47: 75, 48: 190,
    49: 75, 50: 255, 51: 50, 52: 255, 53: 90, 54: 190, 55: 75, 56: 190,
    57: 75, 58: 190, 59: 75, 60: 255, 61: 120, 62: 45, 63: 200, 64: 100,
    65: 50, 66: 180, 67: 90, 68: 45, 69: 255, 70: 120, 71: 45, 72: 190,
    73: 60, 74: 255, 75: 120, 76: 45, 77: 190, 78: 60, 79: 190, 80: 75,
    81: 190, 82: 60, 83: 45, 84: 190, 85: 45, 86: 190, 87: 75, 88: 190,
    89: 75, 90: 190, 91: 60, 92: 190, 93: 90, 94: 45, 95: 45, 96: 190,
    97: 75, 98: 225, 99: 60, 100: 190, 101: 60, 102: 90, 103: 45, 104: 190,
    105: 75, 106: 45, 107: 45, 108: 45, 109: 190, 110: 60, 111: 120, 112: 60,
    113: 30, 114: 45, 115: 45, 116: 225, 117: 75, 118: 225, 119: 60, 120: 225,
    121: 60, 122: 45, 123: 45, 124: 45, 125: 45, 126: 45, 127: 45, 128: 45,
    129: 255, 130: 45, 131: 45, 132: 35, 133: 45, 134: 45, 135: 45, 136: 45,
    137: 45, 138: 45, 139: 45, 140: 45, 141: 45, 142: 45, 143: 25, 144: 3,
    145: 3, 146: 3, 147: 45, 148: 45, 149: 45, 150: 3, 151: 45, 152: 45,
    153: 45, 154: 45, 155: 45, 156: 45, 157: 45, 158: 45, 159: 45, 160: 45,
    161: 255, 162: 90, 163: 255, 164: 90, 165: 255, 166: 90, 167: 255, 168: 90,
    169: 90, 170: 190, 171: 75, 172: 190, 173: 150, 174: 170, 175: 190, 176: 75,
    177: 190, 178: 75, 179: 235, 180: 120, 181: 45, 182: 45, 183: 190, 184: 75,
    185: 65, 186: 45, 187: 255, 188: 120, 189: 45, 190: 45, 191: 235, 192: 120,
    193: 75, 194: 255, 195: 90, 196: 45, 197: 45, 198: 30, 199: 70, 200: 45,
    201: 225, 202: 45, 203: 60, 204: 190, 205: 75, 206: 190, 207: 60, 208: 25,
    209: 190, 210: 75, 211: 45, 212: 25, 213: 190, 214: 45, 215: 60, 216: 120,
    217: 60, 218: 190, 219: 75, 220: 225, 221: 75, 222: 60, 223: 190, 224: 75,
    225: 45, 226: 25, 227: 25, 228: 120, 229: 45, 230: 45, 231: 120, 232: 60,
    233: 45, 234: 45, 235: 45, 236: 75, 237: 45, 238: 45, 239: 45, 240: 45,
    241: 45, 242: 30, 243: 3, 244: 3, 245: 3, 246: 45, 247: 45, 248: 45,
    249: 3, 250: 3, 251: 45, 277: 45, 278: 45, 279: 45, 280: 45, 281: 45,
    282: 45, 283: 45, 284: 45, 285: 45, 286: 255, 287: 127, 288: 255, 289: 90,
    290: 255, 291: 120, 292: 45, 293: 120, 294: 45, 295: 255, 296: 120, 297: 45,
    298: 255, 299: 120, 300: 45, 301: 255, 302: 120, 303: 45, 304: 200, 305: 45,
    306: 255, 307: 90, 308: 255, 309: 190, 310: 45, 311: 200, 312: 75, 313: 125,
    314: 60, 315: 255, 316: 60, 317: 200, 318: 255, 319: 90, 320: 255, 321: 90,
    322: 45, 323: 190, 324: 75, 325: 225, 326: 205, 327: 155, 328: 255, 329: 60,
    330: 225, 331: 60, 332: 255, 333: 120, 334: 45, 335: 180, 336: 200, 337: 120,
    338: 45, 339: 255, 340: 150, 341: 255, 342: 120, 343: 45, 344: 190, 345: 60,
    346: 190, 347: 75, 348: 45, 349: 45, 350: 150, 351: 255, 352: 60, 353: 200,
    354: 200, 355: 45, 356: 180, 357: 90, 358: 255, 359: 45, 360: 125, 361: 190,
    362: 90, 363: 150, 364: 255, 365: 120, 366: 45, 367: 225, 368: 75, 369: 200,
    370: 190, 371: 120, 372: 45, 373: 255, 374: 60, 375: 60, 376: 30, 377: 225,
    378: 45, 379: 90, 380: 90, 381: 25, 382: 180, 383: 90, 384: 45, 385: 45,
    386: 150, 387: 150, 388: 45, 389: 45, 390: 45, 391: 45, 392: 235, 393: 120,
    394: 45, 395: 45, 396: 45, 397: 45, 398: 3, 399: 3, 400: 3, 401: 3,
    402: 3, 403: 3, 404: 5, 405: 5, 406: 3, 407: 3, 408: 3, 409: 3,
    410: 3, 411: 45,
}

BALL_MULTIPLIERS_X10: dict[str, int] = {
    "MASTER_BALL": 10, "ULTRA_BALL": 20, "GREAT_BALL": 15,
    "POKE_BALL": 10, "SAFARI_BALL": 15, "NET_BALL": 10,
    "DIVE_BALL": 10, "NEST_BALL": 10, "REPEAT_BALL": 10,
    "TIMER_BALL": 10, "LUXURY_BALL": 10, "PREMIER_BALL": 10,
}
BALL_COSTS: dict[str, int] = {
    "POKE_BALL": 200, "GREAT_BALL": 600, "ULTRA_BALL": 1200,
    "NET_BALL": 1000, "DIVE_BALL": 1000, "NEST_BALL": 1000,
    "REPEAT_BALL": 1000, "TIMER_BALL": 1000, "LUXURY_BALL": 1000,
    "PREMIER_BALL": 200, "SAFARI_BALL": 17, "MASTER_BALL": 1_000_000,
}


@dataclass(frozen=True)
class CaptureContext:
    """Visible capture state plus optional species rate and safety estimates."""

    species: int
    max_hp: int
    current_hp: int
    types: tuple[int, ...] = ()
    level: int = 1
    status: int = 0
    turn_count: int = 0
    already_caught: bool = False
    underwater: bool = False
    catch_rate: int | None = None
    is_trainer_battle: bool = False
    safari_catch_factor: int | None = None
    failure_faint_probability: float = 0.0
    failure_risk_cost: float = 10_000.0

    def __post_init__(self) -> None:
        if self.species <= 0:
            raise ValueError("A capture requires a positive ROM-internal species ID")
        if self.max_hp <= 0 or not 1 <= self.current_hp <= self.max_hp:
            raise ValueError("Only a living Pokemon with valid HP can be captured")
        if not 1 <= self.level <= 100 or self.turn_count < 0:
            raise ValueError("Pokemon level or battle turn count is invalid")
        if self.status < 0 or self.status > 0xFFFFFFFF:
            raise ValueError("Status must be a uint32 RAM status word")
        if len(self.types) > 2 or any(not 0 <= type_id < 18 for type_id in self.types):
            raise ValueError("Capture target has invalid generation-three type IDs")
        if self.catch_rate is not None and not 1 <= self.catch_rate <= 255:
            raise ValueError("Species catch rate must lie in [1, 255]")
        if self.safari_catch_factor is not None and not 1 <= self.safari_catch_factor <= 20:
            raise ValueError("Safari catch factor must lie in [1, 20]")
        if (
            not math.isfinite(self.failure_faint_probability)
            or not 0.0 <= self.failure_faint_probability <= 1.0
            or not math.isfinite(self.failure_risk_cost)
            or self.failure_risk_cost < 0.0
        ):
            raise ValueError("Capture risk estimates must be finite and nonnegative")


@dataclass(frozen=True)
class BallInventory:
    """A checked snapshot; consuming a ball creates a new snapshot."""

    quantities: Mapping[str, int]
    costs: Mapping[str, int] = field(default_factory=lambda: dict(BALL_COSTS))

    def __post_init__(self) -> None:
        for ball, quantity in self.quantities.items():
            if ball not in BALL_MULTIPLIERS_X10 or not isinstance(quantity, int) or quantity < 0:
                raise ValueError("Unknown ball or negative inventory count")
            if ball not in self.costs or not isinstance(self.costs[ball], int) or self.costs[ball] < 0:
                raise ValueError("Available balls need nonnegative economic costs")
        object.__setattr__(self, "quantities", MappingProxyType(dict(self.quantities)))
        object.__setattr__(self, "costs", MappingProxyType(dict(self.costs)))

    def consume(self, ball: str) -> BallInventory:
        if self.quantities.get(ball, 0) <= 0:
            raise ValueError("Cannot consume a ball absent from inventory")
        quantities = dict(self.quantities)
        quantities[ball] -= 1
        return BallInventory(quantities, dict(self.costs))


@dataclass(frozen=True)
class CaptureArithmetic:
    """Exact integer values and rational four-shake probability."""

    catch_value: int
    shake_threshold: int
    probability_fraction: Fraction

    @property
    def probability(self) -> float:
        return float(self.probability_fraction)


@dataclass(frozen=True)
class BallChoice:
    ball: str
    probability: float
    expected_cost_per_capture: float
    risk_adjusted_cost: float
    catch_value: int
    shake_threshold: int
    available_count: int
    finite_inventory_capture_probability: float


@dataclass(frozen=True)
class CapturePreparation:
    """A feasible action predicted without writing enemy HP or status.

    The caller must establish legality and a non-KO prediction. Risk is an
    estimate, not a formal survival claim. Safe weakening should pass
    ``faint_probability=0`` only when justified by the damage verifier.
    """

    action: str
    target_hp: int
    target_status: int
    faint_probability: float
    cost: float = 0.0


@dataclass(frozen=True)
class CaptureDecision:
    action: str
    reason: str
    ball_choice: BallChoice | None = None
    estimated_faint_probability: float = 0.0


class EpistemicCaptureOptimizer:
    """Capture planning with contextual balls, finite inventory, and risk costs."""

    BALL_MULTIPLIERS = {ball: value / 10.0 for ball, value in BALL_MULTIPLIERS_X10.items()}
    BALL_MULTIPLIERS_X10 = BALL_MULTIPLIERS_X10
    BALL_COSTS = BALL_COSTS

    def __init__(self, species_catch_rates: Mapping[int, int] | None = None) -> None:
        self.species_catch_rates = dict(
            SPECIES_CATCH_RATES if species_catch_rates is None else species_catch_rates
        )
        if any(species <= 0 or not 1 <= rate <= 255 for species, rate in self.species_catch_rates.items()):
            raise ValueError("Species rate table contains an invalid internal ID or catch rate")

    @staticmethod
    def compute_capture_arithmetic(
        max_hp: int,
        curr_hp: int,
        catch_rate: int,
        ball_multiplier_x10: int,
        status_mult_x10: int = 10,
        *,
        master_ball: bool = False,
    ) -> CaptureArithmetic:
        """Replicate integer operations, retaining exact shake probability.

        The game's odds=0 division has no meaningful probabilistic model; a zero
        odds value is represented as impossible rather than clamped to odds=1.
        Status words are validated by the contextual API; this primitive accepts
        the exact supported neutral/sleep/freeze/major-status multipliers.
        """
        if max_hp <= 0 or not 1 <= curr_hp <= max_hp:
            raise ValueError("Capture HP must describe a living target")
        if not 1 <= catch_rate <= 255 or not 1 <= ball_multiplier_x10 <= 255:
            raise ValueError("Capture rate and ball multiplier are outside uint8 domains")
        if status_mult_x10 not in (10, 15, 20, 30):
            raise ValueError("Unsupported generation-three status multiplier")
        boosted_rate = catch_rate * ball_multiplier_x10 // 10
        catch_value = boosted_rate * (3 * max_hp - 2 * curr_hp) // (3 * max_hp)
        catch_value = catch_value * status_mult_x10 // 10
        if master_ball or catch_value >= 255:
            return CaptureArithmetic(catch_value, 65536, Fraction(1))
        if catch_value == 0:
            return CaptureArithmetic(0, 0, Fraction(0))
        denominator = math.isqrt(math.isqrt(16711680 // catch_value))
        shake_threshold = min(65536, 1048560 // denominator)
        return CaptureArithmetic(
            catch_value, shake_threshold, Fraction(shake_threshold, 65536) ** 4
        )

    @staticmethod
    def compute_exact_gen3_catch_prob(
        max_hp: int,
        curr_hp: int,
        catch_rate: int,
        ball_mult: float,
        status_mult_x10: int,
    ) -> float:
        """Preserve the PDF API; reject multipliers that cannot be represented."""
        if not math.isfinite(ball_mult) or ball_mult <= 0.0:
            raise ValueError("Ball multiplier must be finite and positive")
        scaled = ball_mult * 10.0
        if not math.isclose(scaled, round(scaled), abs_tol=1e-10):
            raise ValueError("Gen-III ball multipliers must have exact tenths")
        return EpistemicCaptureOptimizer.compute_capture_arithmetic(
            max_hp, curr_hp, catch_rate, round(scaled), status_mult_x10
        ).probability

    @staticmethod
    def calculate_capture_probability(
        max_hp: int,
        current_hp: int,
        catch_rate: int,
        ball_multiplier: int = 10,
        status_bonus: int = 10,
    ) -> float:
        return EpistemicCaptureOptimizer.compute_capture_arithmetic(
            max_hp, current_hp, catch_rate, ball_multiplier, status_bonus
        ).probability

    @staticmethod
    def status_multiplier(status_word: int) -> int:
        if not 0 <= status_word <= 0xFFFFFFFF:
            raise ValueError("Status must be a uint32 value")
        multiplier = 20 if status_word & (0x7 | 0x20) else 10
        # Original code uses independent ifs. Invalid combined primary states
        # consequently retain their native arithmetic rather than an elif.
        if status_word & (0x8 | 0x10 | 0x40 | 0x80):
            multiplier = multiplier * 15 // 10
        return multiplier

    @staticmethod
    def contextual_ball_multiplier(ball: str, context: CaptureContext) -> int:
        if ball not in BALL_MULTIPLIERS_X10:
            raise ValueError("Unknown generation-three ball")
        if ball == "NET_BALL":
            return 30 if 11 in context.types or 6 in context.types else 10
        if ball == "DIVE_BALL":
            return 35 if context.underwater else 10
        if ball == "NEST_BALL":
            return max(10, 40 - context.level)
        if ball == "REPEAT_BALL":
            return 30 if context.already_caught else 10
        if ball == "TIMER_BALL":
            return min(40, context.turn_count + 10)
        return BALL_MULTIPLIERS_X10[ball]

    def capture_arithmetic(self, ball: str, context: CaptureContext) -> CaptureArithmetic:
        catch_rate: int | None
        if ball not in BALL_MULTIPLIERS_X10:
            raise ValueError("Unknown generation-three ball")
        if context.is_trainer_battle:
            return CaptureArithmetic(0, 0, Fraction(0))
        if ball == "SAFARI_BALL" and context.safari_catch_factor is not None:
            catch_rate = context.safari_catch_factor * 1275 // 100
        else:
            catch_rate = context.catch_rate
            if catch_rate is None:
                try:
                    catch_rate = self.species_catch_rates[context.species]
                except KeyError as error:
                    raise ValueError("Unknown ROM-internal species ID; supply its verified catch rate") from error
        return self.compute_capture_arithmetic(
            context.max_hp, context.current_hp, catch_rate,
            self.contextual_ball_multiplier(ball, context), self.status_multiplier(context.status),
            master_ball=ball == "MASTER_BALL",
        )

    def select_ball(self, context: CaptureContext, inventory: BallInventory) -> BallChoice | None:
        """Minimize expected spend plus the explicit cost of a failed throw."""
        choices: list[BallChoice] = []
        for ball, count in inventory.quantities.items():
            if count <= 0:
                continue
            arithmetic = self.capture_arithmetic(ball, context)
            probability = arithmetic.probability
            if probability <= 0.0:
                continue
            cost = inventory.costs[ball] / probability
            risk_cost = (
                context.failure_faint_probability * context.failure_risk_cost
                * (1.0 - probability) / probability
            )
            finite_probability = 1.0 - (1.0 - probability) ** count
            choices.append(
                BallChoice(ball, probability, cost, cost + risk_cost,
                           arithmetic.catch_value, arithmetic.shake_threshold, count,
                           finite_probability)
            )
        if not choices:
            return None
        return min(choices, key=lambda choice: (choice.risk_adjusted_cost, -choice.probability, choice.ball))

    def plan_capture(
        self,
        context: CaptureContext,
        inventory: BallInventory,
        owned_species: set[int] | frozenset[int],
        team_type_coverage_gap: set[int] | frozenset[int],
        preparations: Sequence[CapturePreparation] = (),
    ) -> CaptureDecision:
        """Compare legal preparations against an immediate available throw."""
        if context.is_trainer_battle:
            return CaptureDecision("IGNORE_CAPTURE", "Trainer-owned Pokemon cannot be captured")
        if context.species in owned_species and not set(context.types) & team_type_coverage_gap:
            return CaptureDecision("IGNORE_CAPTURE", "Species is known and adds no requested coverage")
        direct = self.select_ball(context, inventory)
        if direct is None:
            return CaptureDecision("NO_BALL_AVAILABLE", "No stocked ball has a positive capture probability")
        best = CaptureDecision(
            f"THROW_{direct.ball}", "Available ball minimizes estimated cost and failed-throw risk",
            direct, context.failure_faint_probability * (1.0 - direct.probability),
        )
        best_cost = direct.risk_adjusted_cost
        for preparation in preparations:
            if (
                not preparation.action or not math.isfinite(preparation.cost)
                or preparation.cost < 0.0 or not math.isfinite(preparation.faint_probability)
                or not 0.0 <= preparation.faint_probability <= 1.0
            ):
                raise ValueError("Capture preparation needs a legal action and finite risk/cost")
            future = replace(context, current_hp=preparation.target_hp, status=preparation.target_status)
            choice = self.select_ball(future, inventory)
            if choice is None:
                continue
            cost = (
                choice.risk_adjusted_cost + preparation.cost
                + preparation.faint_probability * context.failure_risk_cost
            )
            if cost < best_cost:
                best_cost = cost
                best = CaptureDecision(
                    preparation.action, "Predicted legal preparation improves capture cost after risk",
                    choice, preparation.faint_probability,
                )
        return best

    def decide_capture_action(
        self,
        wild_species: int,
        wild_types: list[int],
        max_hp: int,
        curr_hp: int,
        catch_rate: int,
        status_x10: int,
        owned_species: set[int],
        team_type_coverage_gap: set[int],
    ) -> tuple[str, str]:
        """Preserve the PDF positional API, returning advice rather than executing.

        This legacy signature has no inventory or survival prediction. Use
        ``plan_capture`` in the orchestrator to restrict choices to stocked balls
        and preparations whose safety was actually assessed.
        """
        context = CaptureContext(wild_species, max_hp, curr_hp, tuple(wild_types), catch_rate=catch_rate)
        if wild_species in owned_species and not set(wild_types) & team_type_coverage_gap:
            return "IGNORE_CAPTURE", "Species is known and adds no requested coverage"
        probabilities = {
            ball: self.compute_exact_gen3_catch_prob(
                max_hp, curr_hp, catch_rate, BALL_MULTIPLIERS_X10[ball] / 10.0, status_x10
            ) for ball in ("POKE_BALL", "GREAT_BALL", "ULTRA_BALL")
        }
        if curr_hp / context.max_hp > 0.45 and probabilities["POKE_BALL"] < 0.35:
            if status_x10 == 10:
                return "INDUCE_STATUS_OR_WEAKEN", "Assess legal status or non-KO weakening before capture"
            return "WEAKEN_SAFE", "Assess a non-KO weakening action using the integer damage verifier"
        positive = {ball: probability for ball, probability in probabilities.items() if probability > 0.0}
        if not positive:
            return "INDUCE_STATUS_OR_WEAKEN", "Current integer catch value is zero"
        ball = min(positive, key=lambda name: BALL_COSTS[name] / positive[name])
        return f"THROW_{ball}", f"Estimated capture probability {positive[ball]:.6f}; verify inventory before use"


@dataclass(frozen=True)
class TrainingRoute:
    """An observed route distribution, including encounters and healing travel."""

    name: str
    expected_experience: float
    duration_seconds: float
    faint_probability: float
    travel_seconds: float = 0.0
    healing_seconds: float = 0.0
    reward_variance: float = 0.0
    samples: int = 0

    def __post_init__(self) -> None:
        values = (
            self.expected_experience, self.duration_seconds, self.faint_probability,
            self.travel_seconds, self.healing_seconds, self.reward_variance,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Training route measurements must be finite")
        if (
            self.expected_experience < 0.0 or self.duration_seconds <= 0.0
            or not 0.0 <= self.faint_probability <= 1.0 or self.travel_seconds < 0.0
            or self.healing_seconds < 0.0 or self.reward_variance < 0.0 or self.samples < 0
        ):
            raise ValueError("Training route measurements are outside their domains")


@dataclass(frozen=True)
class RankedTrainingRoute:
    route: TrainingRoute
    xp_per_second: float
    risk_adjusted_score: float
    estimated_faint_probability: float


class MetacognitiveTrainingCurriculum:
    """Select XP/time opportunities while accounting for measured faint risk.

    A route with zero estimated risk is not called proven safe. A Beta posterior
    for observed route outcomes is available to avoid interpreting an untested
    route as having zero risk. When all routes are risky the least-risk route is
    chosen, matching the user's explicit fallback policy.
    """

    def __init__(self, risk_penalty: float = 1000.0, uncertainty_penalty: float = 0.1) -> None:
        if not math.isfinite(risk_penalty) or risk_penalty < 0.0:
            raise ValueError("Risk penalty must be finite and nonnegative")
        if not math.isfinite(uncertainty_penalty) or uncertainty_penalty < 0.0:
            raise ValueError("Uncertainty penalty must be finite and nonnegative")
        self.risk_penalty = risk_penalty
        self.uncertainty_penalty = uncertainty_penalty
        self._outcomes: dict[str, tuple[int, int]] = {}

    def record_outcome(self, route: str, fainted: bool) -> None:
        if not route:
            raise ValueError("Route identifiers cannot be empty")
        faint_count, survival_count = self._outcomes.get(route, (0, 0))
        self._outcomes[route] = (
            faint_count + int(fainted), survival_count + int(not fainted)
        )

    def estimated_risk(self, route: TrainingRoute) -> float:
        if route.name not in self._outcomes:
            return route.faint_probability
        faints, survivals = self._outcomes[route.name]
        posterior_mean = (faints + 1) / (faints + survivals + 2)
        return max(route.faint_probability, posterior_mean)

    def rank_routes(self, routes: Sequence[TrainingRoute]) -> list[RankedTrainingRoute]:
        ranked: list[RankedTrainingRoute] = []
        for route in routes:
            duration = route.duration_seconds + route.travel_seconds + route.healing_seconds
            xp_per_second = route.expected_experience / duration
            risk = self.estimated_risk(route)
            uncertainty = (
                math.sqrt(route.reward_variance / max(1, route.samples)) / duration
            )
            score = (
                xp_per_second * (1.0 - risk) - self.risk_penalty * risk
                - self.uncertainty_penalty * uncertainty
            )
            ranked.append(RankedTrainingRoute(route, xp_per_second, score, risk))
        return sorted(ranked, key=lambda item: (-item.risk_adjusted_score, item.route.name))

    def select_route(self, routes: Sequence[TrainingRoute]) -> RankedTrainingRoute:
        ranked = self.rank_routes(routes)
        if not ranked:
            raise ValueError("A training curriculum needs at least one feasible route")
        if all(item.estimated_faint_probability > 0.0 for item in ranked):
            return min(
                ranked, key=lambda item: (item.estimated_faint_probability, -item.risk_adjusted_score)
            )
        return ranked[0]
