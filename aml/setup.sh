#!/usr/bin/env bash
# One-time Azure Machine Learning setup. Run from the repo root (Codespace or Cloud Shell)
# after `az login`:
#
#   bash aml/setup.sh <github-owner>/<repo>
#
# Creates (safe to rerun): resource group, Azure ML workspace, CPU cluster,
# the paddy-env environment, the paddy-raw data asset (if data/raw exists),
# and a managed identity that GitHub Actions logs in as through OIDC.
# Prints the variables to add to GitHub.
set -euo pipefail

REPO="${1:?usage: bash aml/setup.sh <owner/repo>}"
RG="${RG:-rg-paddy-mlops}"
WS="${WS:-mlw-paddy}"
ID_NAME="${ID_NAME:-id-paddy-github}"

az extension add -n ml --upgrade -y -o none
echo "== Subscription: $(az account show --query name -o tsv)"
for ns in Microsoft.MachineLearningServices Microsoft.Storage Microsoft.KeyVault Microsoft.Insights \
          Microsoft.ContainerRegistry Microsoft.OperationalInsights Microsoft.ManagedIdentity \
          Microsoft.Cdn Microsoft.PolicyInsights Microsoft.Network Microsoft.Compute; do
  az provider register -n "$ns" --wait -o none
done

# Azure for Students only allows a few regions; read them from the policy if there is one.
if [ -z "${LOCATION:-}" ]; then
  ALLOWED=$(az policy assignment list --query "[?parameters.listOfAllowedLocations].parameters.listOfAllowedLocations.value[]" -o tsv 2>/dev/null || true)
  for cand in southeastasia centralindia eastasia uaenorth koreacentral japaneast westeurope northeurope eastus; do
    if [ -z "$ALLOWED" ] || echo "$ALLOWED" | grep -qx "$cand"; then LOCATION="$cand"; break; fi
  done
  LOCATION="${LOCATION:-$(echo "$ALLOWED" | head -1)}"
fi
echo "== Region: $LOCATION   (override: LOCATION=<region> bash aml/setup.sh ...)"

az group create -n "$RG" -l "$LOCATION" -o none
echo "== Workspace (creates storage, key vault, app insights; takes a few minutes)"
az ml workspace show -n "$WS" -g "$RG" -o none 2>/dev/null || az ml workspace create -n "$WS" -g "$RG" -l "$LOCATION" -o none
az configure --defaults group="$RG" workspace="$WS" location="$LOCATION"

echo "== Compute cluster (scales to 0 when idle)"
az ml compute show -n train-cluster -o none 2>/dev/null || az ml compute create -f aml/compute.yml -o none

echo "== Environment paddy-env (image builds on first use, ~10 min)"
az ml environment create -f aml/environment.yml -o none

if [ -d data/raw/dcs ] && [ -n "$(ls -A data/raw/weather 2>/dev/null)" ]; then
  echo "== Data asset paddy-raw from data/raw"
  az ml data create --name paddy-raw --version "$(date +%Y%m%d%H%M)" --type uri_folder --path data/raw -o none
else
  echo "== data/raw is empty: download it first (see the runbook), then rerun this script"
fi

echo "== Identity for GitHub Actions (OIDC, no stored password)"
az identity create -n "$ID_NAME" -g "$RG" -l "$LOCATION" -o none
CLIENT_ID=$(az identity show -n "$ID_NAME" -g "$RG" --query clientId -o tsv)
PRINCIPAL_ID=$(az identity show -n "$ID_NAME" -g "$RG" --query principalId -o tsv)
RG_ID=$(az group show -n "$RG" --query id -o tsv)
az role assignment create --assignee-object-id "$PRINCIPAL_ID" --assignee-principal-type ServicePrincipal \
  --role Contributor --scope "$RG_ID" -o none 2>/dev/null || echo "   (role assignment exists)"
for SUBJ in "environment:production" "ref:refs/heads/main" "pull_request"; do
  NAME="gh-$(echo "$SUBJ" | tr ':/_' '---')"
  az identity federated-credential create --name "$NAME" --identity-name "$ID_NAME" -g "$RG" \
    --issuer https://token.actions.githubusercontent.com --subject "repo:${REPO}:${SUBJ}" \
    --audiences api://AzureADTokenExchange -o none 2>/dev/null || echo "   ($NAME exists)"
done

cat <<OUT

Done. Azure ML studio: https://ml.azure.com  (workspace $WS)

Add these in GitHub -> Settings -> Secrets and variables -> Actions -> Variables:
  AZURE_CLIENT_ID        ${CLIENT_ID}
  AZURE_TENANT_ID        $(az account show --query tenantId -o tsv)
  AZURE_SUBSCRIPTION_ID  $(az account show --query id -o tsv)
  AZURE_RESOURCE_GROUP   ${RG}
  AZURE_ML_WORKSPACE     ${WS}
OUT
