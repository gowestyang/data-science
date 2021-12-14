# Author: Yang Xi
# Date Created: 2020-08-02

source("utilities_plot_association_rules.r")

dfRules0 <- read.csv('output/df_rules_eclat.csv')
dfCategory <- read.csv('data/category.csv')

dfRulesIn <- dfRules0 %>%
  dplyr::select(RuleLhsDescription=antecedents,
                RuleRhsDescription=consequents,
                RuleConfidence=confidence,
                RuleLeverage=leverage) %>%
  left_join(rename(dfCategory, RuleLhsDescription=item, RuleLhsCategory=category), by="RuleLhsDescription") %>%
  left_join(rename(dfCategory, RuleRhsDescription=item, RuleRhsCategory=category), by = "RuleRhsDescription")

plotItemBasket(dfRulesIn, 'X')
ggsave('output/association_network_plot_X.jpg', width=6, height=4)


plotItemBasket(dfRulesIn, NULL, ruleDepth=30, edgeLabel=NULL, weakAssociation=FALSE, weakStrength=5, seed=2)
ggsave('output/association_network_plot_All.jpg', width=6, height=6)

print('Association Rules Visualized.')
