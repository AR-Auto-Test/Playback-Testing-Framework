package org.opencv.core;

import c.b.a.a.a;

/* loaded from: classes2.dex */
public class CvException extends RuntimeException {
    private static final long serialVersionUID = 1;

    public CvException(String str) {
        super(str);
    }

    @Override // java.lang.Throwable
    public String toString() {
        return a.v(a.x("CvException ["), super.toString(), "]");
    }
}