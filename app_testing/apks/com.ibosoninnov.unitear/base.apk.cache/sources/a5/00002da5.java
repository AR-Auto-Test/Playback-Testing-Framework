package h.a.b;

import java.io.File;
import java.io.FilenameFilter;

/* compiled from: ReLinkerInstance.java */
/* loaded from: classes2.dex */
public class g implements FilenameFilter {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ String f6263a;

    public g(f fVar, String str) {
        this.f6263a = str;
    }

    @Override // java.io.FilenameFilter
    public boolean accept(File file, String str) {
        return str.startsWith(this.f6263a);
    }
}