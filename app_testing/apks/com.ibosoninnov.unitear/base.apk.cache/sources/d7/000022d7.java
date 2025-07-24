package com.google.common.flogger.parameter;

import c.b.a.a.a;
import com.google.common.flogger.backend.FormatOptions;

/* loaded from: classes.dex */
public abstract class Parameter {
    private final int index;
    private final FormatOptions options;

    public Parameter(FormatOptions formatOptions, int i) {
        if (formatOptions == null) {
            throw new IllegalArgumentException("format options cannot be null");
        }
        if (i >= 0) {
            this.index = i;
            this.options = formatOptions;
            return;
        }
        throw new IllegalArgumentException(a.j("invalid index: ", i));
    }

    public abstract void accept(ParameterVisitor parameterVisitor, Object obj);

    public final void accept(ParameterVisitor parameterVisitor, Object[] objArr) {
        if (getIndex() < objArr.length) {
            Object obj = objArr[getIndex()];
            if (obj != null) {
                accept(parameterVisitor, obj);
                return;
            } else {
                parameterVisitor.visitNull();
                return;
            }
        }
        parameterVisitor.visitMissing();
    }

    public abstract String getFormat();

    public final FormatOptions getFormatOptions() {
        return this.options;
    }

    public final int getIndex() {
        return this.index;
    }
}