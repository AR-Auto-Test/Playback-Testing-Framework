package com.google.ar.core.dependencies;

import android.os.IBinder;
import android.os.IInterface;
import android.os.Parcel;

/* compiled from: BaseProxy.java */
/* loaded from: classes.dex */
public class d implements IInterface {

    /* renamed from: a  reason: collision with root package name */
    private final IBinder f5575a;

    /* renamed from: b  reason: collision with root package name */
    private final String f5576b;

    public d(IBinder iBinder, String str) {
        this.f5575a = iBinder;
        this.f5576b = str;
    }

    public final Parcel a() {
        Parcel obtain = Parcel.obtain();
        obtain.writeInterfaceToken(this.f5576b);
        return obtain;
    }

    @Override // android.os.IInterface
    public final IBinder asBinder() {
        return this.f5575a;
    }

    public final Parcel b(int i, Parcel parcel) {
        Parcel obtain = Parcel.obtain();
        try {
            try {
                this.f5575a.transact(i, parcel, obtain, 0);
                obtain.readException();
                return obtain;
            } catch (RuntimeException e2) {
                obtain.recycle();
                throw e2;
            }
        } finally {
            parcel.recycle();
        }
    }

    public final void c(int i, Parcel parcel) {
        try {
            this.f5575a.transact(i, parcel, null, 1);
        } finally {
            parcel.recycle();
        }
    }
}