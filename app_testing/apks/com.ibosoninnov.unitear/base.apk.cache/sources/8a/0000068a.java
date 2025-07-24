package c.a.a.x.b;

import android.annotation.TargetApi;
import android.graphics.Matrix;
import android.graphics.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.ListIterator;

/* compiled from: MergePathsContent.java */
@TargetApi(19)
/* loaded from: classes.dex */
public class l implements m, j {

    /* renamed from: a  reason: collision with root package name */
    public final Path f3182a = new Path();

    /* renamed from: b  reason: collision with root package name */
    public final Path f3183b = new Path();

    /* renamed from: c  reason: collision with root package name */
    public final Path f3184c = new Path();

    /* renamed from: d  reason: collision with root package name */
    public final List<m> f3185d = new ArrayList();

    /* renamed from: e  reason: collision with root package name */
    public final c.a.a.z.k.g f3186e;

    public l(c.a.a.z.k.g gVar) {
        this.f3186e = gVar;
    }

    @TargetApi(19)
    public final void a(Path.Op op) {
        Matrix matrix;
        Matrix matrix2;
        this.f3183b.reset();
        this.f3182a.reset();
        for (int size = this.f3185d.size() - 1; size >= 1; size--) {
            m mVar = this.f3185d.get(size);
            if (mVar instanceof d) {
                d dVar = (d) mVar;
                List<m> e2 = dVar.e();
                for (int size2 = e2.size() - 1; size2 >= 0; size2--) {
                    Path g2 = e2.get(size2).g();
                    c.a.a.x.c.o oVar = dVar.k;
                    if (oVar != null) {
                        matrix2 = oVar.e();
                    } else {
                        dVar.f3152c.reset();
                        matrix2 = dVar.f3152c;
                    }
                    g2.transform(matrix2);
                    this.f3183b.addPath(g2);
                }
            } else {
                this.f3183b.addPath(mVar.g());
            }
        }
        m mVar2 = this.f3185d.get(0);
        if (mVar2 instanceof d) {
            d dVar2 = (d) mVar2;
            List<m> e3 = dVar2.e();
            for (int i = 0; i < e3.size(); i++) {
                Path g3 = e3.get(i).g();
                c.a.a.x.c.o oVar2 = dVar2.k;
                if (oVar2 != null) {
                    matrix = oVar2.e();
                } else {
                    dVar2.f3152c.reset();
                    matrix = dVar2.f3152c;
                }
                g3.transform(matrix);
                this.f3182a.addPath(g3);
            }
        } else {
            this.f3182a.set(mVar2.g());
        }
        this.f3184c.op(this.f3182a, this.f3183b, op);
    }

    @Override // c.a.a.x.b.c
    public void b(List<c> list, List<c> list2) {
        for (int i = 0; i < this.f3185d.size(); i++) {
            this.f3185d.get(i).b(list, list2);
        }
    }

    @Override // c.a.a.x.b.j
    public void e(ListIterator<c> listIterator) {
        while (listIterator.hasPrevious() && listIterator.previous() != this) {
        }
        while (listIterator.hasPrevious()) {
            c previous = listIterator.previous();
            if (previous instanceof m) {
                this.f3185d.add((m) previous);
                listIterator.remove();
            }
        }
    }

    @Override // c.a.a.x.b.m
    public Path g() {
        this.f3184c.reset();
        c.a.a.z.k.g gVar = this.f3186e;
        if (gVar.f3331c) {
            return this.f3184c;
        }
        int ordinal = gVar.f3330b.ordinal();
        if (ordinal == 0) {
            for (int i = 0; i < this.f3185d.size(); i++) {
                this.f3184c.addPath(this.f3185d.get(i).g());
            }
        } else if (ordinal == 1) {
            a(Path.Op.UNION);
        } else if (ordinal == 2) {
            a(Path.Op.REVERSE_DIFFERENCE);
        } else if (ordinal == 3) {
            a(Path.Op.INTERSECT);
        } else if (ordinal == 4) {
            a(Path.Op.XOR);
        }
        return this.f3184c;
    }
}