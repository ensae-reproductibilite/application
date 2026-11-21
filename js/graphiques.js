// Contenu du bandeau : tableau des ventes et bloc d'estimation.
// Les bibliothèques d'Observable (`Inputs`, `Plot`) sont passées en argument depuis la page.

const euros = (x) => `${Math.round(x).toLocaleString("fr-FR")} €`;

// Tableau des ventes d'une parcelle ; la ligne sélectionnée est la valeur de l'élément
// (la première par défaut). Sans vente, un élément vide de valeur null.
export function tableVentes(Inputs, ventes) {
  if (!ventes.length) {
    const vide = document.createElement("p");
    vide.textContent = "Aucune vente sur cette parcelle.";
    vide.value = null;
    return vide;
  }
  return Inputs.table(ventes, {
    multiple: false, required: true, value: ventes[0], rows: 6, maxHeight: 220,
    columns: ["date_mutation", "type_local", "surface", "pieces", "valeur", "prix_m2"],
    header: {
      date_mutation: "Date", type_local: "Type", surface: "Surface", pieces: "Pièces",
      valeur: "Valeur", prix_m2: "€/m²",
    },
    format: {
      surface: (x) => `${x} m²`, valeur: euros, prix_m2: euros,
    },
  });
}

// Prix réel et prix estimé côte à côte, avec leur écart
export function blocEstimation(Plot, vente, prediction) {
  const bloc = document.createElement("div");
  if (!prediction || prediction.enCours || prediction.annule) {
    bloc.textContent = "Estimation en cours…";
    return bloc;
  }
  if (prediction.erreur) {
    bloc.innerHTML = `<p class="erreur"><strong>Erreur</strong> : ${prediction.erreur}</p>`;
    return bloc;
  }
  const ecart = prediction.prix_estime - vente.valeur;
  const signe = ecart > 0 ? "+" : "";
  const donnees = [
    { libelle: "Prix de vente", prix: vente.valeur },
    { libelle: "Estimation du modèle", prix: prediction.prix_estime },
  ];
  bloc.innerHTML = `<p><strong>Estimation</strong> : ${euros(prediction.prix_estime)}
    (${euros(prediction.prix_m2)}/m²)<br><strong>Prix de vente</strong> : ${euros(vente.valeur)}
    (${euros(vente.prix_m2)}/m²)<br>Écart : ${signe}${euros(ecart)}
    (${signe}${Math.round((100 * ecart) / vente.valeur)} %)</p>`;
  bloc.append(
    Plot.plot({
      width: 380, height: 110, marginLeft: 130,
      x: { label: "Prix (€)", tickFormat: "s" },
      y: { label: null },
      marks: [
        Plot.barX(donnees, { x: "prix", y: "libelle", fill: "libelle" }),
        Plot.ruleX([0]),
      ],
      color: { range: ["#9ecae1", "#bd0026"], domain: ["Prix de vente", "Estimation du modèle"] },
    })
  );
  return bloc;
}

// Deux histogrammes du prix au m² (commune puis section de la parcelle), avec le prix
// de vente et, si disponible, l'estimation du modèle. `distributions` = {commune: {nom, prix}, section: {code, prix}}
export function histogrammesPrix(Plot, distributions, vente, prediction) {
  const div = document.createElement("div");
  const estime = prediction && prediction.prix_m2 ? prediction.prix_m2 : null;
  const reperes = [{ prix: vente.prix_m2, nom: "Prix de vente", couleur: "#9ecae1" }];
  if (estime) reperes.push({ prix: estime, nom: "Estimation du modèle", couleur: "#bd0026" });
  const histogramme = (titre, valeurs) =>
    Plot.plot({
      title: titre, width: 440, height: 140, marginLeft: 40,
      x: { label: "Prix au m² (€)", tickFormat: "s" },
      y: { label: null, ticks: 3 },
      marks: [
        Plot.rectY(valeurs, Plot.binX({ y: "count" }, { x: (d) => d, fill: "#dcdcdc", thresholds: 30 })),
        Plot.ruleX(reperes, { x: "prix", stroke: "couleur", strokeWidth: 2.5 }),
        Plot.ruleY([0]),
      ],
    });
  // Une légende commune plutôt que des étiquettes sur les traits, qui se chevauchent
  // quand les deux prix sont proches
  const legende = document.createElement("p");
  legende.className = "legende-histos";
  legende.innerHTML = reperes
    .map((r) => `<span style="color: ${r.couleur}">▍</span>${r.nom} (${euros(r.prix)}/m²)`)
    .join(" &nbsp; ");
  div.append(
    legende,
    histogramme(`Prix au m² à ${distributions.commune.nom}`, distributions.commune.prix),
    histogramme(`Prix au m² dans la section ${distributions.section.code}`, distributions.section.prix)
  );
  return div;
}
