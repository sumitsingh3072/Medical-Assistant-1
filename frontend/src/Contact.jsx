import React from 'react';
import { Input } from "./components/ui/input";
import { Textarea } from "./components/ui/textarea";
import { Button } from "./components/ui/button";
import { Card, CardContent } from "./components/ui/card";

const Contact = () => {
  return (
    <div className="flex items-center justify-center min-h-screen bg-background px-4">
      <Card className="w-full max-w-xl shadow-xl border border-border">
        <CardContent className="p-6 sm:p-8">
          <h2 className="text-2xl font-bold mb-6 text-foreground text-center">Contact Us</h2>
          <form className="space-y-5">
            <div>
              <label htmlFor="name" className="block text-sm font-medium mb-1 text-foreground">
                Name
              </label>
              <Input id="name" type="text" placeholder="Your Name" />
            </div>
            <div>
              <label htmlFor="email" className="block text-sm font-medium mb-1 text-foreground">
                Email
              </label>
              <Input id="email" type="email" placeholder="you@example.com" />
            </div>
            <div>
              <label htmlFor="message" className="block text-sm font-medium mb-1 text-foreground">
                Message
              </label>
              <Textarea id="message" placeholder="Your message..." rows={5} />
            </div>
            <Button type="submit" className="w-full">
              Send Message
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
};

export default Contact;
