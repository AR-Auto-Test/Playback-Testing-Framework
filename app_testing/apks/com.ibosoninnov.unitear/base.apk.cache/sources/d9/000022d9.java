package com.google.common.flogger.parameter;

import com.google.common.flogger.backend.FormatChar;
import com.google.common.flogger.backend.FormatOptions;
import com.google.common.flogger.util.Checks;
import java.util.Collections;
import java.util.EnumMap;
import java.util.Map;

/* loaded from: classes.dex */
public final class SimpleParameter extends Parameter {
    private static final Map<FormatChar, SimpleParameter[]> DEFAULT_PARAMETERS;
    private static final int MAX_CACHED_PARAMETERS = 10;
    private final FormatChar formatChar;
    private final String formatString;

    static {
        EnumMap enumMap = new EnumMap(FormatChar.class);
        FormatChar[] values = FormatChar.values();
        for (int i = 0; i < 10; i++) {
            FormatChar formatChar = values[i];
            enumMap.put((EnumMap) formatChar, (FormatChar) createParameterArray(formatChar));
        }
        DEFAULT_PARAMETERS = Collections.unmodifiableMap(enumMap);
    }

    private SimpleParameter(int i, FormatChar formatChar, FormatOptions formatOptions) {
        super(formatOptions, i);
        String buildFormatString;
        this.formatChar = (FormatChar) Checks.checkNotNull(formatChar, "format char");
        if (formatOptions.isDefault()) {
            buildFormatString = formatChar.getDefaultFormatString();
        } else {
            buildFormatString = buildFormatString(formatOptions, formatChar);
        }
        this.formatString = buildFormatString;
    }

    public static String buildFormatString(FormatOptions formatOptions, FormatChar formatChar) {
        char c2 = formatChar.getChar();
        if (formatOptions.shouldUpperCase()) {
            c2 = (char) (c2 & 65503);
        }
        StringBuilder appendPrintfOptions = formatOptions.appendPrintfOptions(new StringBuilder("%"));
        appendPrintfOptions.append(c2);
        return appendPrintfOptions.toString();
    }

    private static SimpleParameter[] createParameterArray(FormatChar formatChar) {
        SimpleParameter[] simpleParameterArr = new SimpleParameter[10];
        for (int i = 0; i < 10; i++) {
            simpleParameterArr[i] = new SimpleParameter(i, formatChar, FormatOptions.getDefault());
        }
        return simpleParameterArr;
    }

    public static SimpleParameter of(int i, FormatChar formatChar, FormatOptions formatOptions) {
        if (i < 10 && formatOptions.isDefault()) {
            return DEFAULT_PARAMETERS.get(formatChar)[i];
        }
        return new SimpleParameter(i, formatChar, formatOptions);
    }

    @Override // com.google.common.flogger.parameter.Parameter
    public void accept(ParameterVisitor parameterVisitor, Object obj) {
        parameterVisitor.visit(obj, this.formatChar, getFormatOptions());
    }

    @Override // com.google.common.flogger.parameter.Parameter
    public String getFormat() {
        return this.formatString;
    }
}