/**
 * @file    registry_test.cpp
 * @brief   Minimal test for backend and scenario registries (demo only).
 * @author  Kamil J.
 * @date    2025-07-10
 *
 * Build:   see CMakeLists.txt, target 'registry_test'
 * Usage:   ./registry_test
 */

#include <iostream>

#include "backend/backend_registry.hpp"
#include "scenarios/scenario_registry.hpp"

void force_backend_registration();
void force_scenario_registration();

int main()
{
    force_backend_registration();
    force_scenario_registration();
    std::cout << "[TEST] Registered backends: ";
    for (const auto& b : BackendRegistry::instance().available_backends()) std::cout << b << " ";
    std::cout << std::endl;

    std::cout << "[TEST] Registered scenarios: ";
    for (const auto& s : ScenarioRegistry::instance().available_scenarios()) std::cout << s << " ";
    std::cout << std::endl;

    // Test create backend
    auto cpu = BackendRegistry::instance().create("cpu");
    if (cpu)
    {
        std::cout << "[TEST] Backend 'cpu' created. Name: " << cpu->name() << std::endl;
    }
    else
    {
        std::cout << "[FAIL] Backend 'cpu' NOT created." << std::endl;
    }

    auto cuda = BackendRegistry::instance().create("cuda");
    if (cuda)
    {
        std::cout << "[TEST] Backend 'cuda' created. Name: " << cuda->name() << std::endl;
    }
    else
    {
        std::cout << "[FAIL] Backend 'cuda' NOT created." << std::endl;
    }

    // Test create scenario
    auto dummy = ScenarioRegistry::instance().create("dummy");
    if (dummy)
    {
        std::cout << "[TEST] Scenario 'dummy' created. Name: " << dummy->name() << std::endl;
    }
    else
    {
        std::cout << "[FAIL] Scenario 'dummy' NOT created." << std::endl;
    }

    return 0;
}