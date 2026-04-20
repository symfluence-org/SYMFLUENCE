# SYMFLUENCE Licensing Policy

This document describes how SYMFLUENCE is licensed and how the SYMFLUENCE
Foundation administers licensing for uses that fall outside the open source
license. It is intended for contributors, users, institutional adopters, and
any organization considering building a product or platform that incorporates
SYMFLUENCE.

It should be read alongside [LICENSE](LICENSE), [CLA.md](CLA.md),
[CORPORATE_CLA.md](CORPORATE_CLA.md), and [GOVERNANCE.md](GOVERNANCE.md). If
anything in this document conflicts with those instruments, the instruments
control.

---

## 1. Summary

- SYMFLUENCE is open source under **GPL-3.0-or-later**.
- Anyone may use, modify, and redistribute SYMFLUENCE under those terms.
- Redistribution or derivative works must also be released under GPL-3.0-or-later.
- The SYMFLUENCE Foundation administers **additional licenses** — including
  commercial licenses — for organizations whose use cases are not compatible
  with GPL-3.0-or-later obligations.
- Projects that plan to distribute, operate, or commercialize platforms built
  on SYMFLUENCE are encouraged to engage with the Foundation **during
  scoping**, not after development.

---

## 2. Open Source License

SYMFLUENCE is licensed under the GNU General Public License, version 3 or
later (GPL-3.0-or-later). The full text is in [LICENSE](LICENSE).

GPL-3.0-or-later permits you, at no charge, to:

- Run SYMFLUENCE for any purpose, including commercial purposes.
- Study and modify the source code.
- Redistribute SYMFLUENCE and your modifications.
- Build derivative works.

In exchange, GPL-3.0-or-later requires that if you **distribute** SYMFLUENCE
or a derivative work, you do so under GPL-3.0-or-later, provide source code
to recipients, and preserve the license notices. "Distribute" includes
shipping software to customers, bundling SYMFLUENCE into a product, or
providing SYMFLUENCE as part of a service in ways that trigger the license's
network-use provisions under certain configurations.

If you are running SYMFLUENCE internally within your organization for
research, modeling, or operational purposes — without redistribution — the
GPL-3.0-or-later license imposes no additional obligations on you.

---

## 3. Contributions and the CLA

All contributions to SYMFLUENCE are governed by a Contributor License
Agreement:

- **Individuals** sign the [Individual CLA](CLA.md).
- **Organizations** sign the [Corporate CLA](CORPORATE_CLA.md) to authorize
  contributions by designated employees.

Both CLAs grant the SYMFLUENCE Foundation the rights necessary to license
SYMFLUENCE under GPL-3.0-or-later **and** under additional license terms
([CLA.md](CLA.md) §5; [CORPORATE_CLA.md](CORPORATE_CLA.md) §6). This is the
legal basis for the dual-licensing structure described in Section 4 below.

Contributors retain copyright in their contributions. The CLA is a license
grant, not an assignment.

---

## 4. Dual Licensing

SYMFLUENCE operates under a dual-licensing model:

1. **Open source license (GPL-3.0-or-later).** Available to everyone, under
   the terms described in Section 2.
2. **Additional licenses administered by the Foundation.** Available by
   agreement with the SYMFLUENCE Foundation, for uses not compatible with
   GPL-3.0-or-later obligations.

The second category exists because GPL-3.0-or-later is not a fit for every
organization's needs. Organizations that embed SYMFLUENCE into proprietary
products, operate it as part of a commercial service with licensing or
redistribution constraints, or need indemnification and support terms that
an open source license cannot provide, typically need an additional license.

The Foundation will consider additional license requests on a case-by-case
basis, taking into account the intended use, the organization's contribution
history, the public benefit of the arrangement, and the Foundation's
sustainability.

The dual-licensing structure does **not** reduce the rights available under
GPL-3.0-or-later. Those rights remain available to everyone, including
organizations that also hold additional licenses. Foundation-administered
licenses are about license terms (e.g. proprietary distribution,
indemnification, support obligations), not about access to a parallel
proprietary feature set. SYMFLUENCE is not operated as a "community edition
plus enterprise edition" project: the GPL-3.0-or-later distribution is the
canonical project, and any additional Foundation-administered license is a
re-licensing of that same code under different terms.

---

## 5. Derivative Platforms and Commercial Wrappers

This section applies to any project that builds a **platform, service, or
product that incorporates SYMFLUENCE as a component** — including:

- Cloud-hosted services that expose SYMFLUENCE functionality to users.
- Commercial software products that bundle SYMFLUENCE.
- Managed deployments, operational systems, or tiered service offerings
  built around SYMFLUENCE.
- Platforms that wrap SYMFLUENCE in proprietary orchestration, interface,
  or integration layers.

For all such projects, two things are true:

**First, the GPL-3.0-or-later license already governs the SYMFLUENCE
components used in your platform.** This means your distribution of those
components (and, depending on architecture, your distribution of your
platform as a whole) may be subject to GPL-3.0-or-later obligations. You
should obtain independent legal advice about how these obligations apply to
your specific architecture.

**Second, the Foundation strongly encourages engagement during scoping,
not after development.** Licensing questions that are easy to resolve early
— whether GPL-3.0-or-later compliance is sufficient, whether an additional
license is appropriate, what attribution and naming expectations apply — are
substantially more costly to resolve after a platform has been built,
deployed, or funded.

If your project is in the planning, proposal, or early development stage,
please contact the Foundation at **licensing@symfluence.org** with:

1. A brief description of the intended platform or service.
2. The role SYMFLUENCE plays in the architecture (component, core, wrapped,
   extended).
3. Expected distribution or deployment model (internal, commercial,
   cloud-hosted, bundled).
4. Timeline and any relevant funding or institutional context.

The Foundation will typically respond within two weeks with one of three
outcomes: (a) confirmation that GPL-3.0-or-later compliance is sufficient
for the intended use, (b) a proposal for an additional license arrangement,
or (c) a scoping conversation to work through open questions.

Engagement is not an obligation. GPL-3.0-or-later permits you to build on
SYMFLUENCE without contacting the Foundation. But for projects that intend
to operate commercially, secure institutional funding, or distribute to
customers, early engagement is the norm we expect and the path that produces
the cleanest outcomes.

---

## 6. Attribution

All distributions and derivative works of SYMFLUENCE must preserve copyright
notices and attribute the project as described in the LICENSE file. In
addition, the Foundation asks that platforms, publications, and institutional
reports that rely on SYMFLUENCE:

- **Name SYMFLUENCE explicitly** in technical documentation and published
  descriptions of the system. Referring to SYMFLUENCE as "an underlying
  framework" without naming it is not sufficient.
- **Cite the companion papers** (Eythorsson et al., 2026a, 2026b, 2026c)
  when SYMFLUENCE is used in peer-reviewed research or technical reports
  that describe modeling methodology.
- **Link to the canonical repository** (https://github.com/symfluence-org/SYMFLUENCE)
  in online documentation.

These expectations are not a licensing requirement beyond what
GPL-3.0-or-later imposes. They are a community norm that reflects the
Foundation's position that scientific infrastructure deserves visible
attribution, and that omitting attribution — whether deliberately or by
institutional pattern — erodes the conditions under which open source
scientific software remains sustainable.

---

## 7. The SYMFLUENCE Foundation

Licensing is administered by the SYMFLUENCE Foundation, an independent
nonprofit entity established to steward the SYMFLUENCE framework. The
Foundation holds the rights granted under the CLAs, administers additional
licenses, signs corporate CLAs on behalf of the project, and serves as the
legal counterparty for institutional agreements.

The Foundation is in the process of being formally incorporated. During the
interim period, licensing correspondence should be directed to
**licensing@symfluence.org** and will be handled by the Foundation's founding
steward on behalf of the Foundation.

Licensing decisions, including the granting of additional licenses under
Section 4, are Foundation-scoped decisions made in accordance with the
Foundation's statutes (currently being formalized as part of incorporation)
and administered by designated officers. They are distinct from day-to-day
maintainer and project-governance decisions, which are described in
[GOVERNANCE.md](GOVERNANCE.md) and remain the responsibility of the lead
maintainer and the maintainer group.

---

## 8. Questions

If you are unsure whether your intended use of SYMFLUENCE requires an
additional license, the safest path is to ask. The Foundation would rather
have an early conversation than resolve an ambiguity after the fact.

- **Licensing:** licensing@symfluence.org
- **Contributions and CLAs:** dev@symfluence.org
- **Governance and maintainer questions:** see [GOVERNANCE.md](GOVERNANCE.md)

---

## Document History

| Version | Date       | Summary                                          |
|---------|------------|--------------------------------------------------|
| 1.0     | 2026-04-20 | Initial licensing policy                         |
