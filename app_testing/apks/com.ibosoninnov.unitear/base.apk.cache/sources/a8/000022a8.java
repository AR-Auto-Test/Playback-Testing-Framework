package com.google.common.flogger.backend;

/* JADX WARN: Enum visitor error
jadx.core.utils.exceptions.JadxRuntimeException: Init of enum DECIMAL uses external variables
	at jadx.core.dex.visitors.EnumVisitor.createEnumFieldByConstructor(EnumVisitor.java:444)
	at jadx.core.dex.visitors.EnumVisitor.processEnumFieldByRegister(EnumVisitor.java:391)
	at jadx.core.dex.visitors.EnumVisitor.extractEnumFieldsFromFilledArray(EnumVisitor.java:320)
	at jadx.core.dex.visitors.EnumVisitor.extractEnumFieldsFromInsn(EnumVisitor.java:258)
	at jadx.core.dex.visitors.EnumVisitor.convertToEnum(EnumVisitor.java:151)
	at jadx.core.dex.visitors.EnumVisitor.visit(EnumVisitor.java:100)
 */
/* JADX WARN: Failed to restore enum class, 'enum' modifier and super class removed */
/* loaded from: classes.dex */
public final class FormatChar {
    private static final /* synthetic */ FormatChar[] $VALUES;
    public static final FormatChar BOOLEAN;
    public static final FormatChar CHAR;
    public static final FormatChar DECIMAL;
    public static final FormatChar EXPONENT;
    public static final FormatChar EXPONENT_HEX;
    public static final FormatChar FLOAT;
    public static final FormatChar GENERAL;
    public static final FormatChar HEX;
    private static final FormatChar[] MAP;
    public static final FormatChar OCTAL;
    public static final FormatChar STRING;
    private final int allowedFlags;
    private final String defaultFormatString;
    private final char formatChar;
    private final FormatType type;

    static {
        FormatChar formatChar = new FormatChar("STRING", 0, 's', FormatType.GENERAL, "-#", true);
        STRING = formatChar;
        FormatChar formatChar2 = new FormatChar("BOOLEAN", 1, 'b', FormatType.BOOLEAN, "-", true);
        BOOLEAN = formatChar2;
        FormatChar formatChar3 = new FormatChar("CHAR", 2, 'c', FormatType.CHARACTER, "-", true);
        CHAR = formatChar3;
        FormatType formatType = FormatType.INTEGRAL;
        FormatChar formatChar4 = new FormatChar("DECIMAL", 3, 'd', formatType, "-0+ ,", false);
        DECIMAL = formatChar4;
        FormatChar formatChar5 = new FormatChar("OCTAL", 4, 'o', formatType, "-#0", false);
        OCTAL = formatChar5;
        FormatChar formatChar6 = new FormatChar("HEX", 5, 'x', formatType, "-#0", true);
        HEX = formatChar6;
        FormatType formatType2 = FormatType.FLOAT;
        FormatChar formatChar7 = new FormatChar("FLOAT", 6, 'f', formatType2, "-#0+ ,", false);
        FLOAT = formatChar7;
        FormatChar formatChar8 = new FormatChar("EXPONENT", 7, 'e', formatType2, "-#0+ ", true);
        EXPONENT = formatChar8;
        FormatChar formatChar9 = new FormatChar("GENERAL", 8, 'g', formatType2, "-0+ ,", true);
        GENERAL = formatChar9;
        FormatChar formatChar10 = new FormatChar("EXPONENT_HEX", 9, 'a', formatType2, "-#0+ ", true);
        EXPONENT_HEX = formatChar10;
        $VALUES = new FormatChar[]{formatChar, formatChar2, formatChar3, formatChar4, formatChar5, formatChar6, formatChar7, formatChar8, formatChar9, formatChar10};
        MAP = new FormatChar[26];
        FormatChar[] values = values();
        for (int i = 0; i < 10; i++) {
            FormatChar formatChar11 = values[i];
            MAP[indexOf(formatChar11.getChar())] = formatChar11;
        }
    }

    private FormatChar(String str, int i, char c2, FormatType formatType, String str2, boolean z) {
        this.formatChar = c2;
        this.type = formatType;
        this.allowedFlags = FormatOptions.parseValidFlags(str2, z);
        this.defaultFormatString = "%" + c2;
    }

    private boolean hasUpperCaseVariant() {
        return (this.allowedFlags & 128) != 0;
    }

    private static int indexOf(char c2) {
        return (c2 | ' ') - 97;
    }

    private static boolean isLowerCase(char c2) {
        return (c2 & ' ') != 0;
    }

    public static FormatChar of(char c2) {
        FormatChar formatChar = MAP[indexOf(c2)];
        if (isLowerCase(c2)) {
            return formatChar;
        }
        if (formatChar == null || !formatChar.hasUpperCaseVariant()) {
            return null;
        }
        return formatChar;
    }

    public static FormatChar valueOf(String str) {
        return (FormatChar) Enum.valueOf(FormatChar.class, str);
    }

    public static FormatChar[] values() {
        return (FormatChar[]) $VALUES.clone();
    }

    public int getAllowedFlags() {
        return this.allowedFlags;
    }

    public char getChar() {
        return this.formatChar;
    }

    public String getDefaultFormatString() {
        return this.defaultFormatString;
    }

    public FormatType getType() {
        return this.type;
    }
}