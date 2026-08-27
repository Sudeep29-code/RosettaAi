TRANSLATION_RULES = {

    "python": {
        "java": {
            "def": "public static",
            "True": "true",
            "False": "false",
            "None": "null",
            "print(": "System.out.println("
        },

        "javascript": {
            "def": "function ",
            "True": "true",
            "False": "false",
            "None": "null",
            "print(": "console.log("
        },

        "cpp": {
            "def": "",
            "True": "true",
            "False": "false",
            "None": "nullptr",
            "print(": "cout << "
        }
    },

    "java": {
        "python": {
            "true": "True",
            "false": "False",
            "null": "None",
            "System.out.println(": "print("
        },

        "javascript": {
            "true": "true",
            "false": "false",
            "null": "null",
            "System.out.println(": "console.log("
        },

        "cpp": {
            "true": "true",
            "false": "false",
            "null": "nullptr",
            "System.out.println(": "cout << "
        }
    },

    "javascript": {
        "python": {
            "true": "True",
            "false": "False",
            "null": "None",
            "console.log(": "print("
        },

        "java": {
            "true": "true",
            "false": "false",
            "null": "null",
            "console.log(": "System.out.println("
        },

        "cpp": {
            "true": "true",
            "false": "false",
            "null": "nullptr",
            "console.log(": "cout << "
        }
    },

    "cpp": {
        "python": {
            "true": "True",
            "false": "False",
            "nullptr": "None",
            "cout << ": "print("
        },

        "java": {
            "true": "true",
            "false": "false",
            "nullptr": "null",
            "cout << ": "System.out.println("
        },

        "javascript": {
            "true": "true",
            "false": "false",
            "nullptr": "null",
            "cout << ": "console.log("
        }
    }
}