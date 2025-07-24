package com.google.common.flogger.util;

import c.b.a.a.a;

/* loaded from: classes.dex */
public class Checks {
    private Checks() {
    }

    public static void checkArgument(boolean z, String str) {
        if (!z) {
            throw new IllegalArgumentException(str);
        }
    }

    public static String checkMetadataIdentifier(String str) {
        if (!str.isEmpty()) {
            if (isLetter(str.charAt(0))) {
                for (int i = 1; i < str.length(); i++) {
                    char charAt = str.charAt(i);
                    if (!isLetter(charAt) && ((charAt < '0' || charAt > '9') && charAt != '_')) {
                        throw new IllegalArgumentException(a.q("identifier must contain only ASCII letters, digits or underscore: ", str));
                    }
                }
                return str;
            }
            throw new IllegalArgumentException(a.q("identifier must start with an ASCII letter: ", str));
        }
        throw new IllegalArgumentException("identifier must not be empty");
    }

    public static <T> T checkNotNull(T t, String str) {
        if (t != null) {
            return t;
        }
        throw new NullPointerException(a.q(str, " must not be null"));
    }

    private static boolean isLetter(char c2) {
        return ('a' <= c2 && c2 <= 'z') || ('A' <= c2 && c2 <= 'Z');
    }
}