#!/usr/bin/env bash
# Create the Azure resources for the paddy-yield API. Run in Azure Cloud Shell (Bash).
#
#   bash infra/bootstrap.sh <github-owner>/<repo> ghcr.io/<github-owner>/<repo>:latest
#
# Creates (safe to rerun):
#   resource group, Log Analytics workspace, Container Apps environment,
#   container app running the API image, and a managed identity that GitHub
#   Actions can log in as through OIDC (no stored password).
# Prints the four values to add as GitHub Actions variables.
set -euo pipefail

REPO="${1:?usage: bootstrap.sh <owner/repo> <image>}"
IMAGE="${2:?usage: bootstrap.sh <owner/repo> <image>}"
RG="${RG:-rg-paddy-mlops}"
APP="${APP:-ca-paddy-yield}"
CAE="${CAE:-cae-paddy}"
LAW="${LAW:-law-paddy}"
ID_NAME="${ID_NAME:-id-paddy-github}"

echo "== Subscription: $(az account show --query name -o tsv)"

echo "== Registering resource providers (first time takes a few minutes)"
for ns in Microsoft.App Microsoft.OperationalInsights Microsoft.ManagedIdentity; do
  az provider register -n "$ns" --wait -o none
done

# Azure for Students only allows a handful of regions. Read them from the policy if present.
if [ -z "${LOCATION:-}" ]; then
  ALLOWED=$(az policy assignment list --query "[?contains(displayName,'llowed') && parameters.listOfAllowedLocations].parameters.listOfAllowedLocations.value[]" -o tsv 2>/dev/null || true)
  for cand in southeastasia centralindia eastasia uaenorth koreacentral japaneast westeurope northeurope eastus; do
    if [ -z "$ALLOWED" ] || echo "$ALLOWED" | grep -qx "$cand"; then LOCATION="$cand"; break; fi
  done
  LOCATION="${LOCATION:-$(echo "$ALLOWED" | head -1)}"
fi
echo "== Region: $LOCATION   (override with: LOCATION=<region> bash infra/bootstrap.sh ...)"

az group create -n "$RG" -l "$LOCATION" -o none

echo "== Log Analytics"
az monitor log-analytics workspace create -g "$RG" -n "$LAW" -l "$LOCATION" -o none
WS_ID=$(az monitor log-analytics workspace show -g "$RG" -n "$LAW" --query customerId -o tsv)
WS_KEY=$(az monitor log-analytics workspace get-shared-keys -g "$RG" -n "$LAW" --query primarySharedKey -o tsv)

echo "== Container Apps environment"
if ! az containerapp env show -n "$CAE" -g "$RG" -o none 2>/dev/null; then
  az containerapp env create -n "$CAE" -g "$RG" -l "$LOCATION" \
    --logs-workspace-id "$WS_ID" --logs-workspace-key "$WS_KEY" -o none
fi

echo "== Container app (image must already exist on ghcr.io and be Public)"
if ! az containerapp show -n "$APP" -g "$RG" -o none 2>/dev/null; then
  az containerapp create -n "$APP" -g "$RG" --environment "$CAE" --image "$IMAGE" \
    --target-port 8000 --ingress external --min-replicas 0 --max-replicas 1 \
    --cpu 0.5 --memory 1.0Gi --env-vars MODEL_VERSION=bootstrap -o none
fi
FQDN=$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)

echo "== Managed identity + GitHub OIDC trust"
az identity create -n "$ID_NAME" -g "$RG" -l "$LOCATION" -o none
CLIENT_ID=$(az identity show -n "$ID_NAME" -g "$RG" --query clientId -o tsv)
PRINCIPAL_ID=$(az identity show -n "$ID_NAME" -g "$RG" --query principalId -o tsv)
RG_ID=$(az group show -n "$RG" --query id -o tsv)
az role assignment create --assignee-object-id "$PRINCIPAL_ID" --assignee-principal-type ServicePrincipal \
  --role Contributor --scope "$RG_ID" -o none 2>/dev/null || echo "   (role assignment already exists)"
for SUBJ in "environment:production" "ref:refs/heads/main"; do
  NAME="gh-$(echo "$SUBJ" | tr ':/' '--')"
  az identity federated-credential create --name "$NAME" --identity-name "$ID_NAME" -g "$RG" \
    --issuer https://token.actions.githubusercontent.com --subject "repo:${REPO}:${SUBJ}" \
    --audiences api://AzureADTokenExchange -o none 2>/dev/null || echo "   ($NAME already exists)"
done

cat <<OUT

Done. App URL: https://${FQDN}   (try /health and /docs)

Add these in GitHub -> repo Settings -> Secrets and variables -> Actions -> Variables:
  AZURE_CLIENT_ID             ${CLIENT_ID}
  AZURE_TENANT_ID             $(az account show --query tenantId -o tsv)
  AZURE_SUBSCRIPTION_ID       $(az account show --query id -o tsv)
  LOG_ANALYTICS_WORKSPACE_ID  ${WS_ID}
  AZURE_RESOURCE_GROUP        ${RG}
  AZURE_CONTAINER_APP         ${APP}
OUT
