/**
 * @file    register_scenarios.cpp
 * @brief   Registers all scenarios in the global ScenarioRegistry.
 * @author  Kamil J.
 * @date    2025-07-10
 *
 * Performs static registration of all benchmark scenarios.
 */

#include "dummy_scenario.hpp"
#include "scenarios/scenario_registry.hpp"
// #include "scenarios/ml_batch_scenario.hpp" // Example for more advanced

namespace
{
/**
 * @brief Static block to register all scenarios.
 */
struct StaticScenarioRegistrations
{
    StaticScenarioRegistrations()
    {
        ScenarioRegistry::instance().register_scenario("dummy", [] { return std::make_unique<DummyScenario>(); });
        // ScenarioRegistry::instance().register_scenario("ml_batch", [] {
        //     return std::make_unique<MLBatchScenario>();
        // });
        // Register more scenarios here as needed.
    }
};
static StaticScenarioRegistrations registrations;
}  // namespace

// **Force registration function**
void force_scenario_registration()
{
}