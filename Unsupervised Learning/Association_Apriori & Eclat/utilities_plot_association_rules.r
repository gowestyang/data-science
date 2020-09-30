# Author: Yang Xi
# Date Created: 2019-10-15
# Date Modified: 2020-08-02

library(dplyr)
library(purrr)
library(network)
library(RColorBrewer)
library(GGally)
library(igraph)
library(ggthemes)
library(sna)
options(stringsAsFactors = FALSE)

# Function to extract cycles (loops) out of network graph
FindCycles = function(g) {
  Cycles = NULL
  for(v1 in V(g)) {
    if(igraph::degree(g, v1, mode="in") == 0) { next }
    GoodNeighbors = neighbors(g, v1, mode="out")
    GoodNeighbors = GoodNeighbors[GoodNeighbors > v1]
    for(v2 in GoodNeighbors) {
      TempCyc = lapply(all_simple_paths(g, v2, v1, mode="out"), function(p) c(v1,p))
      TempCyc = TempCyc[sapply(TempCyc, min) == sapply(TempCyc, `[`, 1)]
      Cycles = c(Cycles, TempCyc)
    }
  }
  return(Cycles)
}

FindBigCycles <- function(cycles) {
  bigCycles <- list()
  cyclesRemain <- cycles
  
  while(length(cyclesRemain)>0){
    whichBig <- which.max(sapply(cyclesRemain, length))[1]
    cycleBig <- cyclesRemain[[whichBig]]
    bigCycles <- c(bigCycles, list(cycleBig))
    
    cyclesRemain <- cyclesRemain[-whichBig]
    cyclesRemainInBig <- sapply(cyclesRemain, function(v) sum(sapply(v, function(i) !(i %in% cycleBig)))==0)
    cyclesRemain <- cyclesRemain[!cyclesRemainInBig]
  }
  return(bigCycles)
}

CyclesToLinks <- function(cycles, vertexNames){
  getLinks <- function(cyc){
    for(i in 1:(length(cyc)-1)){
      cyc[i] <- paste(cyc[i], "-", cyc[i+1])
    }
    cyc <- head(cyc, -1)
    return(cyc)
  }
  
  cycles <- lapply(cycles, function(cyc) vertexNames[cyc])
  cycles <- lapply(cycles, getLinks)
}

getCycleGroup <- function(linkCycles){
  df_linkGroups <- data.frame(Link=character(), CycleGroup=numeric())
  for(i in seq_along(linkCycles)){
    cyc <- linkCycles[[i]]
    df_linkGroups <- rbind(df_linkGroups, data.frame(Link=cyc, CycleGroup=i))
  }
  if(length(df_linkGroups)>0){
    df_linkGroups <- df_linkGroups %>%
      group_by(Link) %>%
      top_n(-1, wt=CycleGroup) %>%
      as.data.frame()
  }
  return(df_linkGroups)
}

# Function to generate a basket of a specific item
getBasket <- function(dfPairRules, item, depth){
  l_node <- list(list(parent="", node=item))
  dfRules <- data.frame()
  
  for(i in 1:depth){
    l_child <- list()
    for(node in l_node){
      parent <- node$parent
      item <- node$node
      
      rules <- dfPairRules %>%
        filter(RuleLhsDescription == item,
               RuleRhsDescription != parent) %>%
        group_by(SameCategory) %>%
        top_n(1, wt = RuleLeverage) %>%
        as.data.frame()
      
      dfRules <- rbind(dfRules, rules)
      l_child <- c(l_child, map(rules$RuleRhsDescription, function(x) list(parent=item, node=x)))
    }
    l_node <- l_child
  }
  return(dfRules)
}

# Function to plot rules of a specific item
# dfRulesIn: association rules
# item: identifier of the item. If NULL, will plot top N (N=ruleDepth) rules with highest leverage.
# ruleDepth: if given a specific item, will recursively plot top 1 leverage search to the ruleDepth level.
# edgeLabel: NULL, "Confidence" or "Strength", where "Strength" is measured by leverage.
# weakAssociation: whether to display negative association in dashed line.
# seed: random seed for plotting
plotItemBasket <- function(dfRulesIn, item, ruleDepth=3, edgeLabel=NULL, weakAssociation=FALSE, weakStrength=0, seed=1){
  
  # Define display line type for association rule with leverage < 0
  if(weakAssociation){
    weakLinkType <- 2 # dashed line
  } else {
    weakLinkType <- 0 # no edge line
  }
  
  # Rules to be ploted
  if(is.null(item)){
    dfRules <- dfRulesIn %>%
      top_n(ruleDepth, wt=RuleLeverage) %>%
      mutate(Confidence = paste0(round(RuleConfidence*100),"%"),
             Strength = round(RuleLeverage*100, 2),
             AssociationType = ifelse(Strength>weakStrength, 1, weakLinkType)
             ) %>%
      dplyr::select(RuleLhsDescription, RuleRhsDescription, RuleLhsCategory, RuleRhsCategory, Confidence, Strength, AssociationType) %>%
      as.data.frame()
  } else {
    dfRules <- dfRulesIn %>%
      mutate(SameCategory = RuleLhsCategory==RuleRhsCategory) %>%
      group_by(RuleLhsDescription) %>%
      top_n(2, wt=RuleLeverage) %>%
      getBasket(item, ruleDepth) %>%
      mutate(Confidence = paste0(round(RuleConfidence*100),"%"),
             Strength = round(RuleLeverage*100, 2),
             AssociationType = ifelse(Strength>0, 1, weakLinkType)
             ) %>%
      dplyr::select(RuleLhsDescription, RuleRhsDescription, RuleLhsCategory, RuleRhsCategory, Confidence, Strength, AssociationType) %>%
      as.data.frame()
  }
  
  # Extract category for vortex color
  categoryMap <- rbind(
    dplyr::select(dfRules, Cat=RuleLhsDescription, Category=RuleLhsCategory),
    dplyr::select(dfRules, Cat=RuleRhsDescription, Category=RuleRhsCategory)
  ) %>%
    unique()
  row.names(categoryMap) <- categoryMap$Cat
  
  # Set of rules with positive association
  dfRulesPos <- dfRules %>% filter(Strength > 0)
  
  # Identify big cycles (loops) for edges from positive associated rules
  net <- graph_from_data_frame(dfRulesPos)
  df_linkGroups <- net %>%
    FindCycles() %>%
    FindBigCycles() %>%
    CyclesToLinks(names(V(net))) %>%
    getCycleGroup()
  
  dfRulesCyc <- dfRules
  dfRulesCyc <- dfRulesCyc %>%
    mutate(Link = paste(RuleLhsDescription, "-", RuleRhsDescription)) %>%
    left_join(df_linkGroups, by = "Link") %>%
    mutate(CycleGroup = ifelse(is.na(CycleGroup), 0, CycleGroup)+1) %>%
    dplyr::select(-c(Link, RuleLhsCategory))
  
  # Build and plot network
  netRules <- network(dfRulesCyc, loops=TRUE, matrix.type="edgelist", ignore.eval=FALSE)
  netRules %v% "Category" = categoryMap[netRules %v% "vertex.names",]$Category
  
  set.seed(seed)
  ggnet2(netRules,
         color = "Category", alpha=0.5, size=9, palette="Set2",
         label=TRUE, label.size=4,
         edge.color="CycleGroup", edge.alpha=0.5,
         edge.size=1, edge.lty="AssociationType",
         edge.label=edgeLabel, edge.label.size=3,
         arrow.size=8, arrow.gap=0.05, arrow.type="open",
         layout.exp=0.5)
}










