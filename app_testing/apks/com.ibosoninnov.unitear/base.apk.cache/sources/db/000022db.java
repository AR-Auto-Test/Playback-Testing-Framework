package com.google.common.flogger.parser;

/* loaded from: classes.dex */
public abstract class BraceStyleMessageParser extends MessageParser {
    private static final char BRACE_STYLE_SEPARATOR = ',';

    public static int nextBraceFormatTerm(String str, int i) {
        while (i < str.length()) {
            int i2 = i + 1;
            char charAt = str.charAt(i);
            if (charAt == '{') {
                return i2 - 1;
            }
            if (charAt != '\'') {
                i = i2;
            } else if (i2 != str.length()) {
                i = i2 + 1;
                if (str.charAt(i2) != '\'') {
                    int i3 = i - 2;
                    while (i != str.length()) {
                        int i4 = i + 1;
                        if (str.charAt(i) == '\'') {
                            i = i4;
                        } else {
                            i = i4;
                        }
                    }
                    throw ParseException.withStartPosition("unmatched single quote", str, i3);
                }
                continue;
            } else {
                throw ParseException.withStartPosition("trailing single quote", str, i2 - 1);
            }
        }
        return -1;
    }

    public static void unescapeBraceFormat(StringBuilder sb, String str, int i, int i2) {
        int i3;
        int i4 = i;
        boolean z = false;
        while (true) {
            if (i >= i2) {
                break;
            }
            int i5 = i + 1;
            char charAt = str.charAt(i);
            if (charAt == '\\' || charAt == '\'') {
                int i6 = i5 - 1;
                if (charAt == '\\') {
                    i = i5 + 1;
                    if (str.charAt(i5) != '\'') {
                        continue;
                    }
                } else {
                    i = i5;
                }
                sb.append((CharSequence) str, i4, i6);
                if (i == i2) {
                    i4 = i;
                    break;
                }
                if (z) {
                    i3 = i;
                    z = false;
                } else if (str.charAt(i) != '\'') {
                    z = true;
                    i3 = i;
                } else {
                    i3 = i + 1;
                }
                int i7 = i3;
                i4 = i;
                i = i7;
            } else {
                i = i5;
            }
        }
        if (i4 < i2) {
            sb.append((CharSequence) str, i4, i2);
        }
    }

    public abstract void parseBraceFormatTerm(MessageBuilder<?> messageBuilder, int i, String str, int i2, int i3, int i4);

    @Override // com.google.common.flogger.parser.MessageParser
    public final <T> void parseImpl(MessageBuilder<T> messageBuilder) {
        int i;
        int i2;
        String message = messageBuilder.getMessage();
        int nextBraceFormatTerm = nextBraceFormatTerm(message, 0);
        while (nextBraceFormatTerm >= 0) {
            int i3 = nextBraceFormatTerm + 1;
            int i4 = i3;
            int i5 = 0;
            while (i4 < message.length()) {
                int i6 = i4 + 1;
                char charAt = message.charAt(i4);
                char c2 = (char) (charAt - '0');
                if (c2 < '\n') {
                    i5 = (i5 * 10) + c2;
                    if (i5 >= 1000000) {
                        throw ParseException.withBounds("index too large", message, i3, i6);
                    }
                    i4 = i6;
                } else {
                    int i7 = i6 - 1;
                    int i8 = i7 - i3;
                    if (i8 != 0) {
                        if (message.charAt(i3) == '0' && i8 > 1) {
                            throw ParseException.withBounds("index has leading zero", message, i3, i7);
                        }
                        if (charAt != '}') {
                            if (charAt == ',') {
                                int i9 = i6;
                                while (i9 != message.length()) {
                                    int i10 = i9 + 1;
                                    if (message.charAt(i9) == '}') {
                                        i = i10;
                                        i2 = i6;
                                    } else {
                                        i9 = i10;
                                    }
                                }
                                throw ParseException.withStartPosition("unterminated parameter", message, nextBraceFormatTerm);
                            }
                            throw ParseException.withBounds("malformed index", message, i3, i6);
                        }
                        i2 = -1;
                        i = i6;
                        parseBraceFormatTerm(messageBuilder, i5, message, nextBraceFormatTerm, i2, i);
                        nextBraceFormatTerm = nextBraceFormatTerm(message, i);
                    } else {
                        throw ParseException.withBounds("missing index", message, nextBraceFormatTerm, i6);
                    }
                }
            }
            throw ParseException.withStartPosition("unterminated parameter", message, nextBraceFormatTerm);
        }
    }

    @Override // com.google.common.flogger.parser.MessageParser
    public final void unescape(StringBuilder sb, String str, int i, int i2) {
        unescapeBraceFormat(sb, str, i, i2);
    }
}