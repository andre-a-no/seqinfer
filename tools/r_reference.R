# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
#
# Reference values from independent R implementations, for tests/test_r_reference.py:
#   rpact     group sequential boundaries, futility boundaries, drift (shift), crossing
#             probabilities, expected sample sizes, inflation factors, boundaries at realised
#             information fractions, sample sizes for means, rates and events;
#   survival  log-rank statistics (the score test of a Cox model on counting-process data, so
#             that late entry is handled), at interim analyses of a simulated trial;
#   stats     the two-proportion chi-square test.
#
#   Rscript tools/r_reference.R > tests/r_reference.json
#
# rpact, survival and jsonlite are in Debian and Ubuntu (r-cran-rpact, r-cran-survival,
# r-cran-jsonlite); the CI job "r-reference" installs them and runs the comparison against a
# freshly generated file.
suppressPackageStartupMessages({
  library(rpact)
  library(survival)
  library(jsonlite)
})

spending_spec <- function(type, gamma) {
  if (type %in% c("asOF", "bsOF")) return("obrien-fleming")
  if (type %in% c("asP", "bsP")) return("pocock")
  list("power", gamma)
}

design_case <- function(alpha, sided, fractions, type, gamma = NA, beta_type = "none", gamma_b = NA, power = 0.8) {
  args <- list(kMax = length(fractions), alpha = alpha, beta = 1 - power, sided = sided,
               informationRates = fractions, typeOfDesign = type)
  if (!is.na(gamma)) args$gammaA <- gamma
  if (beta_type != "none") {
    args$typeBetaSpending <- beta_type
    args$bindingFutility <- FALSE
    if (!is.na(gamma_b)) args$gammaB <- gamma_b
  }
  d <- do.call(getDesignGroupSequential, args)
  ch <- getDesignCharacteristics(d)
  futility <- NULL
  if (beta_type != "none") futility <- I(as.numeric(d$futilityBounds))  # I(): an array even of length 1
  list(
    alpha = alpha, sided = sided, fractions = fractions, power = power,
    spending = spending_spec(type, gamma),
    futility_spending = if (beta_type == "none") NULL else spending_spec(beta_type, gamma_b),
    critical = as.numeric(d$criticalValues),
    futility_bounds = futility,
    shift = ch$shift,
    inflation = ch$inflationFactor,
    rejection = as.numeric(ch$rejectionProbabilities),
    futility_probabilities = if (beta_type == "none") NULL else I(as.numeric(ch$futilityProbabilities)),
    asn1 = ch$averageSampleNumber1,
    asn0 = ch$averageSampleNumber0
  )
}

of_spending <- function(alpha, t) 2 * (1 - pnorm(qnorm(1 - alpha / 2) / sqrt(t)))

designs <- list()
add <- function(x) designs[[length(designs) + 1]] <<- x
fraction_sets <- list(c(0.25, 0.5, 0.75, 1), c(0.2, 0.4, 0.6, 0.8, 1), c(0.3, 0.7, 1), c(0.5, 1))
for (fr in fraction_sets) {
  for (sided in c(1, 2)) {
    alpha <- if (sided == 1) 0.025 else 0.05
    add(design_case(alpha, sided, fr, "asOF", power = 0.9))
    add(design_case(alpha, sided, fr, "asP", power = 0.8))
    for (g in c(1, 2, 3)) add(design_case(alpha, sided, fr, "asKD", gamma = g, power = 0.8))
  }
  add(design_case(0.025, 1, fr, "asOF", beta_type = "bsOF", power = 0.9))
  add(design_case(0.025, 1, fr, "asOF", beta_type = "bsP", power = 0.8))
  add(design_case(0.025, 1, fr, "asP", beta_type = "bsKD", gamma_b = 2, power = 0.8))
  add(design_case(0.01, 1, fr, "asKD", gamma = 2, beta_type = "bsKD", gamma_b = 3, power = 0.9))
}

# Boundaries at realised information (Lan-DeMets): planned O'Brien-Fleming spending at 2.5%,
# analyses at other fractions of the planned maximum; the last one is final and spends all.
# Boundaries depend only on ratios of information, so rpact gets the fractions over the last.
realised <- list()
for (fr in list(c(0.3, 0.62, 1), c(0.18, 0.45, 0.8, 1), c(0.3, 0.6, 0.9), c(0.35, 0.7, 1.12), c(0.5, 0.95))) {
  k <- length(fr)
  spend <- c(of_spending(0.025, pmin(fr[-k], 1)), 0.025)
  d <- getDesignGroupSequential(kMax = k, alpha = 0.025, sided = 1, informationRates = fr / fr[k],
                                typeOfDesign = "asUser", userAlphaSpending = spend)
  realised[[length(realised) + 1]] <- list(alpha = 0.025, spending = "obrien-fleming", fractions = fr,
                                           critical = as.numeric(d$criticalValues))
}

# Sample sizes.  allocationRatioPlanned is treatment over control: our allocation r / (1 + r).
plan <- getDesignGroupSequential(kMax = 3, alpha = 0.025, beta = 0.2, sided = 1,
                                 informationRates = c(1 / 3, 2 / 3, 1), typeOfDesign = "asOF")
plan_spec <- list(alpha = 0.025, sided = 1, fractions = c(1 / 3, 2 / 3, 1), spending = "obrien-fleming", power = 0.8)
plan2 <- getDesignGroupSequential(kMax = 4, alpha = 0.05, beta = 0.1, sided = 2,
                                  informationRates = c(0.25, 0.5, 0.75, 1), typeOfDesign = "asP")
plan2_spec <- list(alpha = 0.05, sided = 2, fractions = c(0.25, 0.5, 0.75, 1), spending = "pocock", power = 0.9)
sizes <- list()
for (p in list(list(plan, plan_spec), list(plan2, plan2_spec))) {
  d <- p[[1]]
  spec <- p[[2]]
  for (r in c(1, 2, 0.5)) {
    a <- r / (1 + r)
    for (pp in list(c(0.3, 0.2), c(0.12, 0.1), c(0.5, 0.65))) {
      s <- getSampleSizeRates(d, pi1 = pp[1], pi2 = pp[2], allocationRatioPlanned = r)
      sizes[[length(sizes) + 1]] <- list(design = spec, kind = "rates", p_treatment = pp[1], p_control = pp[2],
                                         allocation = a, n = s$maxNumberOfSubjects)
    }
    for (hr in c(0.7, 0.5, 1.3)) {
      s <- getSampleSizeSurvival(d, hazardRatio = hr, pi2 = 0.2, eventTime = 12, accrualTime = 12,
                                 followUpTime = 12, allocationRatioPlanned = r)
      sizes[[length(sizes) + 1]] <- list(design = spec, kind = "events", hazard_ratio = hr, allocation = a,
                                         n = s$maxNumberOfEvents)
    }
    s <- getSampleSizeMeans(d, alternative = 0.3, stDev = 1.5, groups = 2, normalApproximation = TRUE,
                            allocationRatioPlanned = r)
    sizes[[length(sizes) + 1]] <- list(design = spec, kind = "two_means", effect = 0.3, sigma = 1.5,
                                       allocation = a, n = s$maxNumberOfSubjects)
  }
  s <- getSampleSizeMeans(d, alternative = 0.3, stDev = 1.5, groups = 1, normalApproximation = TRUE)
  sizes[[length(sizes) + 1]] <- list(design = spec, kind = "mean", effect = 0.3, sigma = 1.5, n = s$maxNumberOfSubjects)
}

# Log-rank statistics at the interim analyses of a trial with staggered entry: the score test
# of a Cox model on (entry, exit] data, cut at the time of the k-th event.
set.seed(20261016)
n <- 240
arm <- rep(c(1, 0), n / 2)
entry <- runif(n, 0, 12)
event_time <- entry + rexp(n, 0.05 * ifelse(arm == 1, 0.65, 1))
dropout <- entry + rexp(n, 0.01)
exit <- pmin(event_time, dropout, 36)
status <- as.integer(exit == event_time)
times <- sort(exit[status == 1])
analyses <- list()
for (k in c(30, 60, 90, length(times))) {
  cut <- times[k]
  keep <- entry < cut
  stop_k <- pmin(exit[keep], cut)
  status_k <- as.integer(status[keep] == 1 & exit[keep] <= cut)
  fit <- coxph(Surv(entry[keep], stop_k, status_k) ~ arm[keep], ties = "breslow")
  null_fit <- coxph(Surv(entry[keep], stop_k, status_k) ~ arm[keep], init = 0, iter.max = 0, ties = "breslow")
  u <- sum(residuals(null_fit, type = "score"))
  analyses[[length(analyses) + 1]] <- list(events = k, score = unname(u), score_chisq = unname(fit$score),
                                           log_hazard_ratio = unname(coef(fit)))
}
logrank <- list(arm = arm, entry = entry, exit = exit, status = status, analyses = analyses)

proportions <- list()
for (case in list(c(48, 400, 31, 400), c(12, 90, 25, 110), c(150, 1000, 120, 1000))) {
  t <- prop.test(c(case[1], case[3]), c(case[2], case[4]), correct = FALSE)
  proportions[[length(proportions) + 1]] <- list(successes_treatment = case[1], n_treatment = case[2],
                                                 successes_control = case[3], n_control = case[4],
                                                 chisq = unname(t$statistic))
}

cat(toJSON(list(
  generated_by = "tools/r_reference.R",
  versions = list(R = paste(R.version$major, R.version$minor, sep = "."),
                  rpact = as.character(packageVersion("rpact")),
                  survival = as.character(packageVersion("survival"))),
  designs = designs, realised = realised, sample_sizes = sizes, logrank = logrank, proportions = proportions
), auto_unbox = TRUE, digits = NA, null = "null", pretty = TRUE))
cat("\n")
