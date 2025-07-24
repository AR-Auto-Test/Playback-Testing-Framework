package com.bumptech.glide.load.data;

import android.os.ParcelFileDescriptor;
import android.system.ErrnoException;
import android.system.Os;
import android.system.OsConstants;
import c.c.a.m.u.e;
import java.io.IOException;

/* loaded from: classes.dex */
public final class ParcelFileDescriptorRewinder implements e<ParcelFileDescriptor> {

    /* renamed from: a  reason: collision with root package name */
    public final InternalRewinder f5535a;

    /* loaded from: classes.dex */
    public static final class InternalRewinder {

        /* renamed from: a  reason: collision with root package name */
        public final ParcelFileDescriptor f5536a;

        public InternalRewinder(ParcelFileDescriptor parcelFileDescriptor) {
            this.f5536a = parcelFileDescriptor;
        }

        public ParcelFileDescriptor rewind() {
            try {
                Os.lseek(this.f5536a.getFileDescriptor(), 0L, OsConstants.SEEK_SET);
                return this.f5536a;
            } catch (ErrnoException e2) {
                throw new IOException(e2);
            }
        }
    }

    /* loaded from: classes.dex */
    public static final class a implements e.a<ParcelFileDescriptor> {
        @Override // c.c.a.m.u.e.a
        public Class<ParcelFileDescriptor> a() {
            return ParcelFileDescriptor.class;
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
        /* JADX DEBUG: Return type fixed from 'c.c.a.m.u.e' to match base method */
        @Override // c.c.a.m.u.e.a
        public e<ParcelFileDescriptor> b(ParcelFileDescriptor parcelFileDescriptor) {
            return new ParcelFileDescriptorRewinder(parcelFileDescriptor);
        }
    }

    public ParcelFileDescriptorRewinder(ParcelFileDescriptor parcelFileDescriptor) {
        this.f5535a = new InternalRewinder(parcelFileDescriptor);
    }

    @Override // c.c.a.m.u.e
    public void b() {
    }

    /* JADX DEBUG: Method merged with bridge method */
    @Override // c.c.a.m.u.e
    /* renamed from: c */
    public ParcelFileDescriptor a() {
        return this.f5535a.rewind();
    }
}