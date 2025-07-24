package com.google.common.flogger.parameter;

import com.google.common.flogger.backend.FormatOptions;

/* loaded from: classes.dex */
public final class DateTimeParameter extends Parameter {
    private final DateTimeFormat format;
    private final String formatString;

    private DateTimeParameter(FormatOptions formatOptions, int i, DateTimeFormat dateTimeFormat) {
        super(formatOptions, i);
        this.format = dateTimeFormat;
        StringBuilder appendPrintfOptions = formatOptions.appendPrintfOptions(new StringBuilder("%"));
        appendPrintfOptions.append(formatOptions.shouldUpperCase() ? 'T' : 't');
        appendPrintfOptions.append(dateTimeFormat.getChar());
        this.formatString = appendPrintfOptions.toString();
    }

    public static Parameter of(DateTimeFormat dateTimeFormat, FormatOptions formatOptions, int i) {
        return new DateTimeParameter(formatOptions, i, dateTimeFormat);
    }

    @Override // com.google.common.flogger.parameter.Parameter
    public void accept(ParameterVisitor parameterVisitor, Object obj) {
        parameterVisitor.visitDateTime(obj, this.format, getFormatOptions());
    }

    @Override // com.google.common.flogger.parameter.Parameter
    public String getFormat() {
        return this.formatString;
    }
}