package com.google.common.flogger.backend;

import java.util.HashSet;
import java.util.Set;

/* loaded from: classes.dex */
public class KeyValueFormatter implements KeyValueHandler {
    private static final Set<Class<?>> FUNDAMENTAL_TYPES = new HashSet();
    private static final int NEWLINE_LIMIT = 1000;
    private boolean haveSeenValues;
    private final StringBuilder out;
    private final String prefix;
    private final String suffix;

    public KeyValueFormatter(String str, String str2, StringBuilder sb) {
        Set<Class<?>> set = FUNDAMENTAL_TYPES;
        set.add(Boolean.class);
        set.add(Byte.class);
        set.add(Short.class);
        set.add(Integer.class);
        set.add(Long.class);
        set.add(Float.class);
        set.add(Double.class);
        this.haveSeenValues = false;
        this.prefix = str;
        this.suffix = str2;
        this.out = sb;
    }

    private static void appendEscaped(StringBuilder sb, String str) {
        int i = 0;
        while (true) {
            int nextEscapableChar = nextEscapableChar(str, i);
            if (nextEscapableChar != -1) {
                sb.append((CharSequence) str, i, nextEscapableChar);
                i = nextEscapableChar + 1;
                char charAt = str.charAt(nextEscapableChar);
                if (charAt == '\t') {
                    charAt = 't';
                } else if (charAt == '\n') {
                    charAt = 'n';
                } else if (charAt == '\r') {
                    charAt = 'r';
                } else if (charAt != '\"' && charAt != '\\') {
                    sb.append((char) 65533);
                }
                sb.append("\\");
                sb.append(charAt);
            } else {
                sb.append((CharSequence) str, i, str.length());
                return;
            }
        }
    }

    private static int nextEscapableChar(String str, int i) {
        while (i < str.length()) {
            char charAt = str.charAt(i);
            if (charAt < ' ' || charAt == '\"' || charAt == '\\') {
                return i;
            }
            i++;
        }
        return -1;
    }

    public void done() {
        if (this.haveSeenValues) {
            this.out.append(this.suffix);
        }
    }

    /* JADX DEBUG: Method merged with bridge method */
    @Override // com.google.common.flogger.backend.KeyValueHandler
    public KeyValueFormatter handle(String str, Object obj) {
        char c2 = ' ';
        if (this.haveSeenValues) {
            this.out.append(' ');
        } else {
            if (this.out.length() > 0) {
                StringBuilder sb = this.out;
                sb.append((sb.length() > 1000 || this.out.indexOf("\n") != -1) ? '\n' : '\n');
            }
            this.out.append(this.prefix);
            this.haveSeenValues = true;
        }
        StringBuilder sb2 = this.out;
        sb2.append(str);
        sb2.append('=');
        if (obj == null) {
            this.out.append(true);
        } else if (FUNDAMENTAL_TYPES.contains(obj.getClass())) {
            this.out.append(obj);
        } else {
            this.out.append('\"');
            appendEscaped(this.out, obj.toString());
            this.out.append('\"');
        }
        return this;
    }
}