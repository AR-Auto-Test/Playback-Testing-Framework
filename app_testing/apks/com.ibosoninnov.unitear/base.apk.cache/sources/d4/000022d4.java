package com.google.common.flogger.parameter;

import com.google.common.flogger.backend.FormatChar;
import com.google.common.flogger.backend.FormatOptions;
import com.google.common.flogger.backend.FormatType;
import java.text.FieldPosition;
import java.text.MessageFormat;
import java.util.Calendar;
import java.util.Date;
import java.util.Locale;

/* loaded from: classes.dex */
public class BraceStyleParameter extends Parameter {
    private static final int MAX_CACHED_PARAMETERS = 10;
    private static final FormatOptions WITH_GROUPING = FormatOptions.of(16, -1, -1);
    private static final MessageFormat prototypeMessageFormatter = new MessageFormat("{0}", Locale.ROOT);
    private static final BraceStyleParameter[] DEFAULT_PARAMETERS = new BraceStyleParameter[10];

    static {
        for (int i = 0; i < 10; i++) {
            DEFAULT_PARAMETERS[i] = new BraceStyleParameter(i);
        }
    }

    private BraceStyleParameter(int i) {
        super(FormatOptions.getDefault(), i);
    }

    public static BraceStyleParameter of(int i) {
        if (i < 10) {
            return DEFAULT_PARAMETERS[i];
        }
        return new BraceStyleParameter(i);
    }

    @Override // com.google.common.flogger.parameter.Parameter
    public void accept(ParameterVisitor parameterVisitor, Object obj) {
        if (FormatType.INTEGRAL.canFormat(obj)) {
            parameterVisitor.visit(obj, FormatChar.DECIMAL, WITH_GROUPING);
        } else if (FormatType.FLOAT.canFormat(obj)) {
            parameterVisitor.visit(obj, FormatChar.FLOAT, WITH_GROUPING);
        } else if (obj instanceof Date) {
            parameterVisitor.visitPreformatted(obj, ((MessageFormat) prototypeMessageFormatter.clone()).format(new Object[]{obj}, new StringBuffer(), (FieldPosition) null).toString());
        } else if (obj instanceof Calendar) {
            parameterVisitor.visitDateTime(obj, DateTimeFormat.DATETIME_FULL, getFormatOptions());
        } else {
            parameterVisitor.visit(obj, FormatChar.STRING, getFormatOptions());
        }
    }

    @Override // com.google.common.flogger.parameter.Parameter
    public String getFormat() {
        return "%s";
    }
}