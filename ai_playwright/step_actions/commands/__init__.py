"""
命令模式实现包
"""

# 导入所有命令类，确保它们被注册到CommandFactory
from ai_playwright.step_actions.commands.assertion_commands import *
from ai_playwright.step_actions.commands.file_commands import *
from ai_playwright.step_actions.commands.interaction_commands import *
from ai_playwright.step_actions.commands.misc_commands import *
from ai_playwright.step_actions.commands.navigation_commands import *
from ai_playwright.step_actions.commands.network_commands import *
from ai_playwright.step_actions.commands.storage_commands import *
from ai_playwright.step_actions.commands.wait_commands import *
from ai_playwright.step_actions.commands.window_commands import *
